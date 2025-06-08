#include <unordered_set>
#include <queue>
#include "NeuralNetwork.hpp"
using namespace std;




// NeuralNetwork -----------------------------------------------------------------------------------------------------------------------------------

void NeuralNetwork::eval() {
    evaluating = true;
}

void NeuralNetwork::train() {
    evaluating = false;
}

void NeuralNetwork::setLearningRate(double lr) {
    learningRate = lr;
}

void NeuralNetwork::setInputNodeIds(std::vector<int> inputNodeIds) {
    this->inputNodeIds = std::move(inputNodeIds);
}

void NeuralNetwork::setOutputNodeIds(std::vector<int> outputNodeIds) {
    this->outputNodeIds = std::move(outputNodeIds);
}

vector<int> NeuralNetwork::getInputNodeIds() const {
    return inputNodeIds;
}

vector<int> NeuralNetwork::getOutputNodeIds() const {
    return outputNodeIds;
}

#include <unordered_set>
#include <queue>
#include "NeuralNetwork.hpp"


vector<double> NeuralNetwork::predict(DataInstance instance) {
    // 1) load inputs
    for (auto* n : nodes) {
        n->preActivationValue = 0;
        n->postActivationValue = 0;
    }
    for (size_t i = 0; i < inputNodeIds.size(); ++i) {
        nodes[inputNodeIds[i]]->postActivationValue = instance.x[i];
    }

    queue<int> q;
    unordered_set<int> seen;
    for (int in : inputNodeIds) {
        for (auto& edge : adjacencyList[in]) {
            q.push(edge.second.dest);
        }
    }
    while (!q.empty()) {
        int v = q.front(); q.pop();
        if (!seen.insert(v).second) continue;
        // gather from all parents
        for (int u = 0; u < (int)adjacencyList.size(); ++u) {
            auto it = adjacencyList[u].find(v);
            if (it != adjacencyList[u].end()) {
                visitPredictNeighbor(it->second);
            }
        }
        visitPredictNode(v);
        for (auto& edge : adjacencyList[v]) {
            if (!seen.count(edge.second.dest)) {
                q.push(edge.second.dest);
            }
        }
    }

    vector<double> out;
    out.reserve(outputNodeIds.size());
    for (int o : outputNodeIds) {
        out.push_back(nodes[o]->postActivationValue);
    }

    if (evaluating) {
        flush();
    } else {
        batchSize++;
        contributions.clear();
        contribute(instance.y, out[0]);
    }
    return out;
}

bool NeuralNetwork::contribute(double y, double p) {
    queue<int> q;
    unordered_set<int> seen;

    double dOut = - (y - p) / (p * (1 - p));
    for (int outId : outputNodeIds) {
        visitContributeNode(outId, dOut);
        contributions[outId] = dOut;
        q.push(outId);
        seen.insert(outId);
    }

    while (!q.empty()) {
        int child = q.front(); q.pop();
        double childContr = contributions[child];

        for (int parent = 0; parent < (int)adjacencyList.size(); ++parent) {
            auto it = adjacencyList[parent].find(child);
            if (it == adjacencyList[parent].end()) continue;
            Connection& c = it->second;

            double parentAccum = 0;
            visitContributeNeighbor(c, childContr, parentAccum);

            visitContributeNode(parent, parentAccum);

            if (!seen.count(parent)) {
                contributions[parent] = parentAccum;
                q.push(parent);
                seen.insert(parent);
            }
        }
    }
    return true;
}

bool NeuralNetwork::update() {
    if (batchSize == 0) return false;

    for (int i = 0; i < (int)nodes.size(); ++i) {
        if (find(inputNodeIds.begin(), inputNodeIds.end(), i) == inputNodeIds.end()) {
            NodeInfo* n = nodes[i];
            n->bias -= learningRate * (n->delta / batchSize);
        }
        nodes[i]->delta = 0;
    }

    for (auto& connMap : adjacencyList) {
        for (auto& pr : connMap) {
            Connection& c = pr.second;
            c.weight -= learningRate * (c.delta / batchSize);
            c.delta = 0;
        }
    }

    flush();
    return true;
}





// Feel free to explore the remaining code, but no need to implement past this point

// ----------- YOU DO NOT NEED TO TOUCH THE REMAINING CODE -----------------------------------------------------------------
// ----------- YOU DO NOT NEED TO TOUCH THE REMAINING CODE -----------------------------------------------------------------
// ----------- YOU DO NOT NEED TO TOUCH THE REMAINING CODE -----------------------------------------------------------------
// ----------- YOU DO NOT NEED TO TOUCH THE REMAINING CODE -----------------------------------------------------------------
// ----------- YOU DO NOT NEED TO TOUCH THE REMAINING CODE -----------------------------------------------------------------







// Constructors
NeuralNetwork::NeuralNetwork() : Graph(0) {
    learningRate = 0.1;
    evaluating = false;
    batchSize = 0;
}

NeuralNetwork::NeuralNetwork(int size) : Graph(size) {
    learningRate = 0.1;
    evaluating = false;
    batchSize = 0;
}

NeuralNetwork::NeuralNetwork(string filename) : Graph() {
    // open file
    ifstream fin(filename);

    // error check
    if (fin.fail()) {
        cerr << "Could not open " << filename << " for reading. " << endl;
        exit(1);
    }

    // load network
    loadNetwork(fin);
    learningRate = 0.1;
    evaluating = false;
    batchSize = 0;

    // close file
    fin.close();
}

NeuralNetwork::NeuralNetwork(istream& in) : Graph() {
    loadNetwork(in);
    learningRate = 0.1;
    evaluating = false;
    batchSize = 0;
}

void NeuralNetwork::loadNetwork(istream& in) {
    int numLayers(0), totalNodes(0), numNodes(0), weightModifications(0), biasModifications(0); string activationMethod = "identity";
    string junk;
    in >> numLayers; in >> totalNodes; getline(in, junk);
    if (numLayers <= 1) {
        cerr << "Neural Network must have at least 2 layers, but got " << numLayers << " layers" << endl;
        exit(1);
    }

    // resize network to accomodate expected nodes.
    resize(totalNodes);
    this->size = totalNodes;

    int currentNodeId(0);

    vector<int> previousLayer;
    vector<int> currentLayer;
    for (int i = 0; i < numLayers; i++) {
        currentLayer.clear();
        //  For each layer

        // get nodes for this layer and activation method
        in >> numNodes; in >> activationMethod; getline(in, junk);

        for (int j = 0; j < numNodes; j++) {
            // For every node, add a new node to the network with proper activationMethod
            // initialize bias to 0.
            updateNode(currentNodeId, NodeInfo(activationMethod, 0, 0));
            // This node has an id of currentNodeId
            currentLayer.push_back(currentNodeId++);
        }

        if (i != 0) {
            // There exists a previous layer, now we set out connections
            for (int k = 0; k < previousLayer.size(); k++) {
                for (int w = 0; w < currentLayer.size(); w++) {

                    // Initialize an initial weight of a sample from the standard normal distribution
                    updateConnection(previousLayer.at(k), currentLayer.at(w), sample());
                }
            }
        }

        // Crawl forward.
        previousLayer = currentLayer;
        layers.push_back(currentLayer);
    }
    in >> weightModifications; getline(in, junk);
    int v(0),u(0); double w(0), b(0);

    // load weights by updating connections
    for (int i = 0; i < weightModifications; i++) {
        in >> v; in >> u; in >> w; getline(in , junk);
        updateConnection(v, u, w);
    }

    in >> biasModifications; getline(in , junk);

    // load biases by updating node info
    for (int i = 0; i < biasModifications; i++) {
        in >> v; in >> b; getline(in, junk);
        NodeInfo* thisNode = getNode(v);
        thisNode->bias = b;
    }

    setInputNodeIds(layers.at(0));
    setOutputNodeIds(layers.at(layers.size()-1));
}

void NeuralNetwork::visitPredictNode(int vId) {
    // accumulate bias, and activate
    NodeInfo* v = nodes.at(vId);
    v->preActivationValue += v->bias;
    v->activate();
}

void NeuralNetwork::visitPredictNeighbor(Connection c) {
    NodeInfo* v = nodes.at(c.source);
    NodeInfo* u = nodes.at(c.dest);
    double w = c.weight;
    u->preActivationValue += v->postActivationValue * w;
}

void NeuralNetwork::visitContributeNode(int vId, double& outgoingContribution) {
    NodeInfo* v = nodes.at(vId);
    outgoingContribution *= v->derive();
    
    //contribute bias derivative
    v->delta += outgoingContribution;
}

void NeuralNetwork::visitContributeNeighbor(Connection& c, double& incomingContribution, double& outgoingContribution) {
    NodeInfo* v = nodes.at(c.source);
    // update outgoingContribution
    outgoingContribution += c.weight * incomingContribution;

    // accumulate weight derivative
    c.delta += incomingContribution * v->postActivationValue;
}

void NeuralNetwork::flush() {
    // set every node value to 0 to refresh computation.
    for (int i = 0; i < nodes.size(); i++) {
        nodes.at(i)->postActivationValue = 0;
        nodes.at(i)->preActivationValue = 0;
    }
    contributions.clear();
    batchSize = 0;
}

double NeuralNetwork::assess(string filename) {
    DataLoader dl(filename);
    return assess(dl);
}

double NeuralNetwork::assess(DataLoader dl) {
    bool stateBefore = evaluating;
    evaluating = true;
    double count(0);
    double correct(0);
    vector<double> output;
    for (int i = 0; i < dl.getData().size(); i++) {
        DataInstance di = dl.getData().at(i);
        output = predict(di);
        if (static_cast<int>(round(output.at(0))) == di.y) {
            correct++;
        }
        count++;
    }

    if (dl.getData().empty()) {
        cerr << "Cannot assess accuracy on an empty dataset" << endl;
        exit(1);
    }
    evaluating = stateBefore;
    return correct / count;
}


void NeuralNetwork::saveModel(string filename) {
    ofstream fout(filename);
    
    fout << layers.size() << " " << getNodes().size() << endl;
    for (int i = 0; i < layers.size(); i++) {
        NodeInfo* layerNode = getNodes().at(layers.at(i).at(0));
        string activationType = getActivationIdentifier(layerNode->activationFunction);

        fout << layers.at(i).size() << " " << activationType << endl;
    }

    int numWeights = 0;
    int numBias = 0;
    stringstream weightStream;
    stringstream biasStream;
    for (int i = 0; i < nodes.size(); i++) {
        numBias++;
        biasStream << i << " " << nodes.at(i)->bias << endl;

        for (auto j = adjacencyList.at(i).begin(); j != adjacencyList.at(i).end(); j++) {
            numWeights++;
            weightStream << j->second.source << " " << j->second.dest << " " << j->second.weight << endl;
        }
    }

    fout << numWeights << endl;
    fout << weightStream.str();
    fout << numBias << endl;
    fout << biasStream.str();

    fout.close();


}

ostream& operator<<(ostream& out, const NeuralNetwork& nn) {
    for (int i = 0; i < nn.layers.size(); i++) {
        out << "layer " << i << ": ";
        for (int j = 0; j < nn.layers.at(i).size(); j++) {
            out << nn.layers.at(i).at(j) << " ";
        }
        out << endl;
    }
    // outputs the nn in dot format
    out << static_cast<const Graph&>(nn) << endl;
    return out;
}

