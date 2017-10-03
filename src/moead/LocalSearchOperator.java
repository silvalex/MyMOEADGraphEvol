package moead;

/**
 * Represents an abstract local search operator, to be implemented according to the
 * chosen representation.
 *
 * @author sawczualex
 */
public abstract class LocalSearchOperator {
	/**
	 * Performs local search on the given individual, returning the best
	 * neighbour found.
	 *
	 * @param ind - the original individual
	 * @param init - the main program class
	 * @param problemIndex - the index to the subproblem we are peforming the local search on
	 * @return Mutated individual
	 */
	public abstract Individual doSearch(Individual ind, MOEAD init, int problemIndex);
}
