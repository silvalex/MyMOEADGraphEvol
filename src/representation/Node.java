package representation;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import moead.Service;
import moead.TaxonomyNode;

public class Node implements Cloneable {
	private List<Edge> incomingEdgeList = new ArrayList<Edge>();
	private List<Edge> outgoingEdgeList = new ArrayList<Edge>();
	private Service s;

	public Node(Service service) {
		s = service;
	}

	public List<Edge> getIncomingEdgeList() {
		return incomingEdgeList;
	}

	public List<Edge> getOutgoingEdgeList() {
		return outgoingEdgeList;
	}

	public double[] getQos() {
		return s.qos;
	}

	public Set<String> getInputs() {
		return s.inputs;
	}

	public Set<String> getOutputs() {
		return s.outputs;
	}

	public String getName() {
		return s.name;
	}

	public Node clone() {
		return new Node(s);
	}

	public List<TaxonomyNode> getTaxonomyOutputs() {
		return s.taxonomyOutputs;
	}

	public void setLayer(int layer) {
		s.layer = layer;
	}

	public int getLayer() {
		return s.layer;
	}

	@Override
	public String toString() {
		return s.name;
	}

	@Override
	public int hashCode() {
		return s.name.hashCode();
	}

	@Override
	public boolean equals(Object other) {
		if (other instanceof Node) {
			Node o = (Node) other;
			return s.name.equals(o.s.name);
		}
		else
			return false;
	}
}
