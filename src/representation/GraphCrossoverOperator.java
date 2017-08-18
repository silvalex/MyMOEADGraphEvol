package representation;

import java.util.HashSet;
import java.util.Set;
import moead.CrossoverOperator;
import moead.Individual;
import moead.MOEAD;

public class GraphCrossoverOperator extends CrossoverOperator {

	@Override
	public Individual doCrossover(Individual ind1, Individual ind2, MOEAD init) {
		if (!(ind1 instanceof GraphIndividual) || !(ind2 instanceof GraphIndividual))
			throw new RuntimeException("GraphCrossoverOperator can only work on objects of type GraphIndividual.");
		GraphIndividual t1 = ((GraphIndividual)ind1);
		GraphIndividual t2 = ((GraphIndividual)ind2);

		// Merge graphs
	    GraphIndividual mergedGraph = mergeGraphs(t1, t2, init);
	    // Extract child from merged structure
	    GraphIndividual newG = init.createNewGraph(mergedGraph, init.startServ.clone(), init.endServ.clone(), init.relevant);
	    newG.setInit(init);
	    newG.evaluate();
		return newG;
	}

	private GraphIndividual mergeGraphs(GraphIndividual g1, GraphIndividual g2, MOEAD init) {
		GraphIndividual newG = new GraphIndividual();

		// Merge nodes
		for (Node n: g1.nodeMap.values()) {
			newG.nodeMap.put(n.getName(), n.clone());
		}
		for (Node n: g2.nodeMap.values()) {
			newG.nodeMap.put(n.getName(), n.clone());
		}

		// Merge edges
		Set<Edge> edgesToMerge = new HashSet<Edge>();
		edgesToMerge.addAll(g1.edgeList);
		edgesToMerge.addAll(g2.edgeList);

		for (Edge e : edgesToMerge) {
			Edge newE = new Edge(e.getIntersect());
			Node fromNode = newG.nodeMap.get(e.getFromNode().getName());
			newE.setFromNode(fromNode);
			Node toNode = newG.nodeMap.get(e.getToNode().getName());
			newE.setToNode(toNode);
			newG.edgeList.add(newE);
			fromNode.getOutgoingEdgeList().add(newE);
			toNode.getIncomingEdgeList().add(newE);
		}
		init.removeDanglingNodes(newG);
		g1.nodeMap.clear();
		g1.nodeMap.putAll(newG.nodeMap);
		g1.edgeList.clear();
		g1.edgeList.addAll(newG.edgeList);

		return g1;
	}
}
