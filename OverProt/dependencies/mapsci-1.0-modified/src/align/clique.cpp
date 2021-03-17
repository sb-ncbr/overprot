

#include "clique.h"

#include "params.h"
#include "runs.h"
#include "protein.h"
#include "matrix.h"

#include <vector>
using std::vector;


// creates empty graphs with default dimensions
CliqueFinder::CliqueFinder() : graph(100, 100),
			       tempGraph(100, 100)
{
}


// construct the connectivity graph of the runs; two runs are connected
// if the frobenius norm of their rotation matrices is "small"
//
void CliqueFinder::buildGraph(const RunsList& runs, Protein* prot1, Protein* prot2)
{
  graph.resize(runs.size(), runs.size());
  tempGraph.resize(runs.size(), runs.size());
    
  for (int i = 0; i < graph.rows(); ++i) {
    for (int j = 0; j < graph.cols(); ++j) {
      double norm = frobenius(runs[i].rot, runs[j].rot);      
      if (norm < Params::RUNS_ROT) {
	graph(i, j) = 1;
      }
      else {
	graph(i, j) = 0;
      }
    }
  }
}


// finds the node with the largest weight
int CliqueFinder::findMaxNode(const RunsList& runs)
{
  int n = runs.size();
  vector<int> degree(n);

  // compute the degree of each node
  for (int i = 0; i < n; ++i) {
    degree[i] = 0;
    for (int j = 0; j < n; ++j) {
      if (graph(i, j) == 1)
	degree[i] += runs[j].size;
    }
  }
    
  int node = 0;
  int maxDeg = degree[0];
  for (int i = 1; i < n; ++i) {
    if (degree[i] > maxDeg) {
      maxDeg = degree[i];
      node  = i;
    }
  }
    
  return node;
}



// checks whether the graph is empty
bool CliqueFinder::isEmpty() 
{
  for (int i = 0; i < graph.rows(); ++i) {
    for (int j = 0; j < graph.cols(); ++j) {
      if (graph(i, j) != 0) {
	return false;
      }
    }
  }
  return true;
}
  
  
// finds the clique of consistent runs of max weight
// (i.e. runs whose rotation martiecs are similar)
//
vector<int> CliqueFinder::consistent(const RunsList& runs, Protein* prot1, Protein* prot2)
{
  if (runs.size() == 0) {
    return vector<int>();
  }
    
  // the indicies of consistent runs
  vector<int> clique;

  buildGraph(runs, prot1, prot2);
  bool done = isEmpty();
    
  while (!done) {
      
    int node = findMaxNode(runs);
    clique.push_back(node);
      
    // find the neighbors of the max node
    vector<int> neighbors;
    for (int i = 0; i < graph.rows(); ++i) {
      if ( (graph(node, i) == 1) && (i != node) ) {
	neighbors.push_back(i);
      }
    } 
      
    // clear the graph, but make a temporary copy
    for (int i = 0; i < graph.rows(); ++i) {
      for (int j = 0; j < graph.cols(); ++j) {
	tempGraph(i, j) = graph(i, j);
	graph(i, j) = 0;
      }
    }
      
    // copy the subgraph defined by the neighbors; make sure at leas one pair connected
    done = true;
    for (int i = 0; i < neighbors.size(); ++i) {
      for (int j = 0; j < neighbors.size(); ++j) {
	int r = neighbors[i];
	int c = neighbors[j];
	  
	graph(r, c) = tempGraph(r, c);

	if ( graph(r, c) != 0 ) {
	  done = false;
	}
      }
    }
  }

  return clique;
}
