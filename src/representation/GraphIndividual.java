package representation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import moead.Individual;
import moead.MOEAD;

public class GraphIndividual extends Individual {
	private double availability;
	private double reliability;
	private double time;
	private double cost;

	public Map<String, Node> nodeMap = new HashMap<String, Node>();
	public List<Edge> edgeList = new ArrayList<Edge>();

	private double[] objectives;
	private MOEAD init;

	@Override
	public Individual generateIndividual() {
		GraphIndividual newInd = init.createNewGraph(null, init.startServ.clone(), init.endServ.clone(), init.relevant);
		newInd.setInit(init);
		newInd.evaluate();
		if (MOEAD.dynamicNormalisation)
			newInd.finishCalculatingFitness();
		return newInd;
	}

	@Override
	public Individual clone() {
		GraphIndividual newInd = new GraphIndividual();
		copyTo(newInd);

		newInd.availability = availability;
		newInd.reliability = reliability;
		newInd.time = time;
		newInd.cost = cost;
		newInd.init = init;
		newInd.objectives = new double[objectives.length];

		System.arraycopy(objectives, 0, newInd.objectives, 0, objectives.length);

		return newInd;
	}

	/**
	 * Copies this graph structure to another GraphIndividual object.
	 *
	 * @param other
	 */
    public void copyTo(GraphIndividual other) {
        for (Node n : nodeMap.values()) {
            Node newN = n.clone();
            other.nodeMap.put( newN.getName(), newN );
        }

        for (Edge e: edgeList) {
            Edge newE = new Edge(e.getIntersect());
            other.edgeList.add(newE);
            Node newFromNode = other.nodeMap.get( e.getFromNode().getName() );
            newE.setFromNode( newFromNode );
            newFromNode.getOutgoingEdgeList().add( newE );
            Node newToNode = other.nodeMap.get( e.getToNode().getName() );
            newE.setToNode( newToNode );
            newToNode.getIncomingEdgeList().add( newE );
        }
    }

	@Override
	/**
	 * A graphic representation of this candidate can be generated by saving this description to a .dot file and
	 * running the command "dot -Tpng filename.dot -o filename.png"
	 */
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("digraph g {");
		for(Edge e: edgeList) {
			builder.append(e);
			builder.append("; ");
		}
		builder.append("}");
		return builder.toString();
	}

	@Override
	public double[] getObjectiveValues() {
		return objectives;
	}

	@Override
	public void setObjectiveValues(double[] newObjectives) {
		objectives = newObjectives;
	}

	@Override
	public double getAvailability() {
		return availability;
	}

	@Override
	public double getReliability() {
		return reliability;
	}

	@Override
	public double getTime() {
		return time;
	}

	@Override
	public double getCost() {
		return cost;
	}

	@Override
	public void evaluate() {
		calculateSequenceFitness(init.numLayers, init.endServ);
	}

	public void calculateSequenceFitness(int numLayers, Node end) {
		availability = 1.0;
		reliability = 1.0;
		time = 0.0;
		cost = 0.0;

		for (Node n : nodeMap.values()) {
			double[] qos = n.getQos();
			availability *= qos[MOEAD.AVAILABILITY];
			reliability *= qos[MOEAD.RELIABILITY];
			cost += qos[MOEAD.COST];
		}

		// Calculate longest time
		time = findLongestPath();
		if (!MOEAD.dynamicNormalisation)
			finishCalculatingFitness();
	}

	/**
	 * Uses the Bellman-Ford algorithm with negative weights to find the longest
	 * path in an acyclic directed graph.
	 *
	 * @return longest time overall
	 */
	private double findLongestPath() {
		Map<String, Double> distance = new HashMap<String, Double>();
		Map<String, Node> predecessor = new HashMap<String, Node>();

		// Step 1: initialize graph
		for (Node node : nodeMap.values()) {
			if (node.getName().equals("start"))
				distance.put(node.getName(), 0.0);
			else
				distance.put(node.getName(), Double.POSITIVE_INFINITY);
		}

		// Step 2: relax edges repeatedly
		for (int i = 1; i < nodeMap.size(); i++) {
			for (Edge e : edgeList) {
				if ((distance.get(e.getFromNode().getName()) -
				        e.getToNode().getQos()[MOEAD.TIME])
				        < distance.get(e.getToNode().getName())) {
					distance.put(e.getToNode().getName(), (distance.get(e.getFromNode().getName()) - e.getToNode().getQos()[MOEAD.TIME]));
					predecessor.put(e.getToNode().getName(), e.getFromNode());
				}
			}
		}

		// Now retrieve total cost
		Node pre = predecessor.get("end");
		double totalTime = 0.0;

		while (pre != null) {
			totalTime += pre.getQos()[MOEAD.TIME];
			pre = predecessor.get(pre.getName());
		}

		return totalTime;
	}

   @Override
   public void finishCalculatingFitness() {
	   objectives = calculateFitness(cost, time, availability, reliability);
   }

	public double[] calculateFitness(double c, double t, double a, double r) {
        a = normaliseAvailability(a, init);
        r = normaliseReliability(r, init);
        t = normaliseTime(t, init);
        c = normaliseCost(c, init);

        double[] objectives = new double[2];
        objectives[0] = t + c;
        objectives[1] = a + r;

        return objectives;
	}

	private double normaliseAvailability(double availability, MOEAD init) {
		if (init.maxAvailability - init.minAvailability == 0.0)
			return 1.0;
		else
			return (init.maxAvailability - availability)/(init.maxAvailability - init.minAvailability);
	}

	private double normaliseReliability(double reliability, MOEAD init) {
		if (init.maxReliability- init.minReliability == 0.0)
			return 1.0;
		else
			return (init.maxReliability - reliability)/(init.maxReliability - init.minReliability);
	}

	private double normaliseTime(double time, MOEAD init) {
		if (init.maxTime - init.minTime == 0.0)
			return 1.0;
		else
			return (time - init.minTime)/(init.maxTime - init.minTime);
	}

	private double normaliseCost(double cost, MOEAD init) {
		if (init.maxCost - init.minCost == 0.0)
			return 1.0;
		else
			return (cost - init.minCost)/(init.maxCost - init.minCost);
	}

	@Override
	public void setInit(MOEAD init) {
		this.init = init;
	}

}
