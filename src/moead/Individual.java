package moead;

/**
 * Represents an individual in the population of solutions. A generation method for
 * the given individual must be implemented.
 *
 * @author sawczualex
 */
public abstract class Individual implements Cloneable {
	/**
	 * Randomly generates a new individual.
	 *
	 * @return new individual
	 */
	public abstract Individual generateIndividual();

	/**
	 * Returns the objective values for this individual.
	 *
	 * @return array of objective values
	 */
	public abstract double[] getObjectiveValues();

	/**
	 * Sets the array of objective values according to the argument passed.
	 *
	 * @param newObjectives
	 */
	public abstract void setObjectiveValues(double[] newObjectives);

	/**
	 * Returns the overall availability (raw) for this individual.
	 *
	 * @return availability
	 */
	public abstract double getAvailability();

	/**
	 * Returns the overall reliability (raw) for this individual.
	 *
	 * @return reliability
	 */
	public abstract double getReliability();

	/**
	 * Returns the overall time (raw) for this individual.
	 *
	 * @return time
	 */
	public abstract double getTime();

	/**
	 * Returns the overall cost (raw) for this individual.
	 *
	 * @return cost
	 */
	public abstract double getCost();

	@Override
	/**
	 * {@inheritDoc}
	 */
	public abstract Individual clone();

	/**
	 * Checks whether one individual dominates another (i.e. has equal and/or better
	 * objective values than another, with at least one better value).
	 *
	 * @param other individual.
	 * @return true if this individual dominates the other, false otherwise.
	 */
	public boolean dominates(Individual other) {
		double[] thisObjValues = getObjectiveValues();
		double[] otherObjValues = other.getObjectiveValues();

		boolean equivalent = true;
		boolean higher = false;

		for (int i = 0; i < thisObjValues.length; i++) {
			// Check if this individual has at least equivalent values for all objectives
			if (thisObjValues[i] < otherObjValues[i]) {
				equivalent = false;
				break;
			}
			// Check if this individual has at least one higher value than the other objectives
			if (thisObjValues[i] > otherObjValues[i]) {
				higher = true;
			}
		}
		return equivalent && higher;
	}
	
	/**
	 * Sets the associated init class to the individual, for accessing global values.
	 * 
	 * @param init
	 */
	public abstract void setInit(MOEAD init);
	
	/**
	 * Calculates the overall QoS attributes, normalises them, and calculates objectives.
	 */
	public abstract void evaluate();
	
	/**
	 * Calculates the objective values for this individual.
	 */
	public abstract void finishCalculatingFitness();

	@Override
	/**
	 * {@inheritDoc}
	 */
	public abstract String toString();

}
