package representation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import moead.Individual;
import moead.MOEAD;
import moead.MutationOperator;

public class GraphMutationOperator extends MutationOperator {

	@Override
	public Individual mutate(Individual ind, MOEAD init) {
		if (!(ind instanceof GraphIndividual))
			throw new RuntimeException("GraphMutationOperator can only work on objects of type GraphIndividual.");
		GraphIndividual graph = (GraphIndividual) ind;


		Object[] nodes = graph.nodeMap.values().toArray();

        // Select node from which to perform mutation
        Node selected = null;
        while (selected == null) {
            Node temp = (Node) nodes[init.random.nextInt( nodes.length )];
            if (!temp.getName().equals( "end" )) {
                selected = temp;
            }
        }

        if (selected.getName().equals( "start" )) {
            // Create an entirely new graph
            graph = init.createNewGraph( null, init.startServ.clone(), init.endServ.clone(), init.relevant );
        }
        else {

            // Find all nodes that should be removed
            Node newEnd   = init.endServ.clone();
            Set<Node> nodesToRemove = findNodesToRemove(selected);
            Set<Edge> edgesToRemove = new HashSet<Edge>();

            // Remove nodes and edges
            for (Node node : nodesToRemove) {
                graph.nodeMap.remove( node.getName() );

                for (Edge e : node.getIncomingEdgeList()) {
                    edgesToRemove.add( e );
                    e.getFromNode().getOutgoingEdgeList().remove( e );
                }
                for (Edge e : node.getOutgoingEdgeList()) {
                    edgesToRemove.add( e );
                    e.getToNode().getIncomingEdgeList().remove( e );
                }
            }

            for (Edge edge : edgesToRemove) {
                graph.edgeList.remove( edge );
            }


            // Create data structures
            Set<Node> unused = new HashSet<Node>(init.relevant);
            Set<Node> relevant = init.relevant;
            Set<String> currentEndInputs = new HashSet<String>();
            Set<Node> seenNodes = new HashSet<Node>();
            List<Node> candidateList = new ArrayList<Node>();

            for (Node node: graph.nodeMap.values()) {
                unused.remove( node );
                seenNodes.add( node );
            }

            // Must add all nodes as seen before adding candidate list entries
            for (Node node: graph.nodeMap.values()) {
                if (!node.getName().equals( "end" ))
                    init.addToCandidateList( node, seenNodes, relevant, candidateList);
            }

            // Update currentEndInputs
            for (Node node : graph.nodeMap.values()) {
                for (String o : node.getOutputs()) {
                    currentEndInputs.addAll(init.taxonomyMap.get(o).endNodeInputs);
                }
            }


            Collections.shuffle(candidateList, init.random);
            Map<String,Edge> connections = new HashMap<String,Edge>();

            // Continue constructing graph
            init.finishConstructingGraph( currentEndInputs, newEnd, candidateList, connections, graph, null, seenNodes, relevant, true );

        }

        graph.evaluate();
		return graph;
	}

	private Set<Node> findNodesToRemove(Node selected) {
	    Set<Node> nodes = new HashSet<Node>();
	    _findNodesToRemove(selected, nodes);
	    return nodes;

	}

	private void _findNodesToRemove(Node current, Set<Node> nodes) {
        nodes.add( current );
        for (Edge e: current.getOutgoingEdgeList()) {
            _findNodesToRemove(e.getToNode(), nodes);
        }
	}

}
