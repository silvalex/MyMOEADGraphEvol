package moead;

import static java.lang.Math.abs;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.w3c.dom.Text;
import org.xml.sax.SAXException;

import representation.GraphCrossoverOperator;
import representation.Edge;
import representation.GraphIndividual;
import representation.GraphLocalSearchOperator;
import representation.GraphMutationOperator;
import representation.Node;

/**
 * The entry class for the program. The main algorithm is executed once MOEAD is instantiated.
 *
 * @author sawczualex
 */
public class MOEAD {
	// Parameter settings
	public static long seed = 1;
	public static int generations = 51;
	public static int popSize = 500;
	public static int numObjectives = 2;
	public static int numNeighbours = 30;
	public static double crossoverProbability = 0.8;
	public static double mutationProbability = 0.1;
	public static double localSearchProbability = 0.1;
	public static StoppingCriteria stopCrit = new GenerationStoppingCriteria(generations);
	public static Individual indType = new GraphIndividual();
	public static MutationOperator mutOperator = new GraphMutationOperator();
	public static CrossoverOperator crossOperator = new GraphCrossoverOperator();
	public static LocalSearchOperator localOperator = new GraphLocalSearchOperator();
	public static int numLocalSearchTries = 30;
	public static String outFileName = "out.stat";
	public static String frontFileName = "front.stat";
	public static String serviceRepository = "services-output.xml";
	public static String serviceTaxonomy = "taxonomy.xml";
	public static String serviceTask = "problem.xml";
	public static boolean tchebycheff = true;
	public static boolean dynamicNormalisation = false;
	// Fitness weights
	public static double w1 = 0.25;
	public static double w2 = 0.25;
	public static double w3 = 0.25;
	public static double w4 = 0.25;

	// Constants
	public static final int AVAILABILITY = 0;
	public static final int RELIABILITY = 1;
	public static final int TIME = 2;
	public static final int COST = 3;

	public static final int CROSSOVER = 0;
	public static final int MUTATION = 1;
	public static final int LOCAL_SEARCH = 2;

	// Internal state
	private Individual[] population = new Individual[popSize];
	private double[][] weights = new double[popSize][numObjectives];
	private double idealPoint[];
	private int[][] neighbourhood = new int[popSize][numNeighbours];
	private Map<String, Node> serviceMap = new HashMap<String, Node>();
	public Map<String, TaxonomyNode> taxonomyMap = new HashMap<String, TaxonomyNode>();
	public Set<String> taskInput;
	public Set<String> taskOutput;
	public Node startServ;
	public Set<Node> relevant;
	public List<Node> relevantList;
	public Node endServ;
	public Random random;
	public int numLayers;
	// Normalisation bounds
	public double minAvailability = 0.0;
	public double maxAvailability = -1.0;
	public double minReliability = 0.0;
	public double maxReliability = -1.0;
	public double minTime = Double.MAX_VALUE;
	public double maxTime = -1.0;
	public double minCost = Double.MAX_VALUE;
	public double maxCost = -1.0;

	// Statistics
	private long[] breedingTime = new long[generations];
	private long[] evaluationTime = new long[generations];
	FileWriter outWriter;
	FileWriter frontWriter;

	/**
	 * Loads and parses the parameter file.
	 *
	 * @param fileName
	 */
	private void parseParamsFile(String fileName) {
		try {
			Scanner scan = new Scanner(new File(fileName));
			while (scan.hasNext()) {
				setParam(scan.next(), scan.next());
			}
			scan.close();
		}
		catch (FileNotFoundException e) {
			System.err.println("Cannot read parameter file.");
			e.printStackTrace();
		}
	}

	/**
	 * Sets the next parameter from the file, checking against a list
	 * of possibilities.
	 *
	 * @param scan
	 */
	private void setParam(String token, String param) {
		try {
			switch(token) {
		        case "seed":
		       	 seed = Long.valueOf(param);
		            break;
		        case "generations":
		       	 generations = Integer.valueOf(param);
		       	 break;
		        case "popSize":
		       	 popSize = Integer.valueOf(param);
		       	 break;
		        case "numObjectives":
		       	 numObjectives = Integer.valueOf(param);
		            break;
		        case "numNeighbours":
		       	 numNeighbours = Integer.valueOf(param);
		            break;
		        case "crossoverProbability":
		       	 crossoverProbability = Double.valueOf(param);
		       	 break;
		        case "mutationProbability":
		       	 mutationProbability = Double.valueOf(param);
		            break;
				case "localSearchProbability":
					localSearchProbability = Double.valueOf(param);
					break;
		        case "stopCrit":
		       	 stopCrit = (StoppingCriteria) Class.forName(param).getConstructor(Integer.TYPE).newInstance(generations);
		       	 break;
		        case "indType":
		       	 indType = (Individual) Class.forName(param).newInstance();
		       	 break;
		        case "mutOperator":
		       	 mutOperator = (MutationOperator) Class.forName(param).newInstance();
		       	 break;
		        case "crossOperator":
		       	 crossOperator = (CrossoverOperator) Class.forName(param).newInstance();
		       	 break;
				case "localOperator":
					localOperator = (LocalSearchOperator) Class.forName(param).newInstance();
					break;
				case "numLocalSearchTries":
					numLocalSearchTries = Integer.valueOf(param);
					break;
		        case "outFileName":
		       	 outFileName = param;
		       	 break;
		        case "frontFileName":
		       	 frontFileName = param;
		       	 break;
		        case "serviceRepository":
		       	 serviceRepository = param;
		       	 break;
		        case "serviceTaxonomy":
		       	 serviceTaxonomy = param;
		       	 break;
		        case "serviceTask":
		       	 serviceTask = param;
		       	 break;
		        case "tchebycheff":
		         tchebycheff = Boolean.valueOf(param);
		         break;
		        case "dynamicNormalisation":
		         dynamicNormalisation = Boolean.valueOf(param);
		         break;
		        case "w1":
		       	 w1 = Double.valueOf(param);
		       	 break;
		        case "w2":
		       	 w2 = Double.valueOf(param);
		       	 break;
		        case "w3":
		       	 w3 = Double.valueOf(param);
		       	 break;
		        case "w4":
		       	 w4 = Double.valueOf(param);
		       	 break;
		        default:
		            throw new IllegalArgumentException("Invalid parameter: " + token);
			}
		}
		catch (InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException | NoSuchMethodException | SecurityException | ClassNotFoundException e) {
			System.err.println("Cannot parse parameter correctly: " + token);
			e.printStackTrace();
		}
	}

	public MOEAD(String[] args) {
		parseParamsFile(args[0]);

		// Read in any additional parameters
		for (int i = 1; i < args.length; i +=2) {
			setParam(args[i], args[i+1]);
		}

		int generation = 0;
		indType.setInit(this);

		// Initialise
		long startTime = System.currentTimeMillis();
		Set<Individual> externalPopulation = new HashSet<Individual>();

		initialise();
		breedingTime[generation] = System.currentTimeMillis() - startTime;

		// While stopping criteria not met
		while(!stopCrit.stoppingCriteriaMet()) {
			startTime = System.currentTimeMillis();
			// Create an array to hold the new generation
			Individual[] newGeneration = new Individual[popSize];
			System.arraycopy(population, 0, newGeneration, 0, popSize);

			// Evolve new individuals for each problem
			for (int i = 0; i < popSize; i++) {
				Individual oldInd = population[i];
				Individual newInd = evolveNewIndividual(oldInd, i, random);

				// Update individual itself
				double oldScore;
				if (tchebycheff)
					oldScore = calculateTchebycheffScore(population[i], i);
				else
					oldScore = calculateScore(population[i], i);

				double newScore;
				if (tchebycheff)
					newScore = calculateTchebycheffScore(newInd, i);
				else
					newScore= calculateScore(newInd, i);
				if (newScore < oldScore) {
					// Replace neighbour with new solution in new generation
					newGeneration[i] = newInd;
				}

				// Update neighbours
				updateNeighbours(newInd, i, newGeneration);
				// Update reference points
				if (tchebycheff)
					updateReference(newInd);
			}
			// If using dynamic normalisation, finish evaluating the population
			if (dynamicNormalisation)
				finishEvaluating();
			// Copy the next generation over as the new population
			population = newGeneration;
			// Update the external population
			Collections.addAll(externalPopulation, population);
			long endTime = System.currentTimeMillis();
			evaluationTime[generation] = endTime - startTime;
			// Write out stats
			writeOutStatistics(outWriter, generation);
			generation++;
		}

		// Write the front to disk
		externalPopulation = produceParetoFront(externalPopulation);
		writeFrontStatistics(frontWriter, externalPopulation);

		// Close writers
		try {
			outWriter.close();
			frontWriter.close();
		}
		catch (IOException e) {
			System.err.println("Cannot close stat writers.");
			e.printStackTrace();
		}

		System.out.println("Done!");

	}

	/**
	 * Creates weight vectors, calculates neighbourhoods, and generates an initial population.
	 */
	public void initialise() {
		// Create stat writers
		try {
			File outFile = new File(outFileName);
			File frontFile = new File(frontFileName);
			outWriter = new FileWriter(outFile);
			frontWriter = new FileWriter(frontFile);
		}
		catch (IOException e) {
			System.err.println("Cannot create stat writers.");
			e.printStackTrace();
		}
		// Parse dataset files
		parseWSCServiceFile(serviceRepository);
		parseWSCTaskFile(serviceTask);
		parseWSCTaxonomyFile(serviceTaxonomy);

		findConceptsForInstances();

		double[] mockQos = new double[4];
		mockQos[TIME] = 0;
		mockQos[COST] = 0;
		mockQos[AVAILABILITY] = 1;
		mockQos[RELIABILITY] = 1;
		Set<String> startOutput = new HashSet<String>();
		startOutput.addAll(taskInput);
		startServ = new Node(new Service("start", mockQos, new HashSet<String>(), taskInput));
		endServ = new Node(new Service("end", mockQos, taskOutput ,new HashSet<String>()));

		populateTaxonomyTree();
		relevant = getRelevantServices(serviceMap, taskInput, taskOutput);
		relevantList = new ArrayList<Node>(relevant);

		if (!dynamicNormalisation)
			calculateNormalisationBounds(relevant);

		// Ensure that mutation and crossover probabilities add up to 1
		if (mutationProbability + crossoverProbability + localSearchProbability != 1.0)
			throw new RuntimeException("The probabilities for mutation, crossover, and local search should add up to 1.");
		// Initialise random number
		random = new Random(seed);
		// Initialise the reference point
		if (tchebycheff)
			initIdealPoint();
		// Create a set of uniformly spread weight vectors
		initWeights();
		// Identify the neighbouring weights for each vector
		identifyNeighbourWeights();
		// Generate an initial population
		for (int i = 0; i < population.length; i++)
			population[i] = indType.generateIndividual();
	}

	/**
	 * Initialises the uniformely spread weight vectors. This code come from the authors'
	 * original code base.
	 */
	private void initWeights() {
		for (int i = 0; i < popSize; i++) {
			if (numObjectives == 2) {
				double[] weightVector = new double[2];
				weightVector[0] = i / (double) popSize;
				weightVector[1] = (popSize - i) / (double) popSize;
				weights[i] = weightVector;
			}
			else if (numObjectives == 3) {
				for (int j = 0; j < popSize; j++) {
					if (i + j < popSize) {
						int k = popSize - i - j;
						double[] weightVector = new double[3];
						weightVector[0] = i / (double) popSize;
						weightVector[1] = j / (double) popSize;
						weightVector[2] = k / (double) popSize;
						weights[i] = weightVector;
					}
				}
			}
			else {
				throw new RuntimeException("Unsupported number of objectives. Should be 2 or 3.");
			}
		}
	}

	/**
	 * Initialises the ideal point used for the Tchebycheff calculation.
	 */
	private void initIdealPoint() {
		idealPoint = new double[numObjectives];
		for (int i = 0; i < numObjectives; i++) {
			idealPoint[i] = 0.0;
		}
	}

	/**
	 * Create a neighbourhood for each weight vector, based on the Euclidean distance between each two vectors.
	 */
	private void identifyNeighbourWeights() {
		// Calculate distance between vectors
		double[][] distanceMatrix = new double[popSize][popSize];

		for (int i = 0; i < popSize; i++) {
			for (int j = 0; j < popSize; j++) {
				if (i != j)
					distanceMatrix[i][j] = calculateDistance(weights[i], weights[j]);
			}
		}

		// Use this information to build the neighbourhood
		for (int i = 0; i < popSize; i++) {
			int[] neighbours = identifyNearestNeighbours(distanceMatrix[i], i);
			neighbourhood[i] = neighbours;
		}
	}

	/**
	 * Calculates the Euclidean distance between two weight vectors.
	 *
	 * @param vector1
	 * @param vector2
	 * @return distance
	 */
	private double calculateDistance(double[] vector1, double[] vector2) {
		double sum = 0;
		for (int i = 0; i < vector1.length; i++) {
			sum += Math.pow((vector1[i] - vector2[i]), 2);
		}
		return Math.sqrt(sum);
	}

	/**
	 * Returns the indices for the nearest neighbours, according to their distance
	 * from the current vector.
	 *
	 * @param distances - a list of distances from the other vectors
	 * @param currentIndex - the index of the current vector
	 * @return indices of nearest neighbours
	 */
	private int[] identifyNearestNeighbours(double[] distances, int currentIndex) {
		Queue<IndexDistancePair> indexDistancePairs = new LinkedList<IndexDistancePair>();

		// Convert the vector of distances to a list of index-distance pairs.
		for(int i = 0; i < distances.length; i++) {
			indexDistancePairs.add(new IndexDistancePair(i, distances[i]));
		}
		// Sort the pairs according to the distance, from lowest to highest.
		Collections.sort((LinkedList<IndexDistancePair>) indexDistancePairs);

		// Get the indices for the required number of neighbours
		int[] neighbours = new int[numNeighbours];

		// Get the neighbours, excluding the vector itself
		IndexDistancePair neighbourCandidate;
		for (int i = 0; i < numNeighbours; i++) {
			neighbourCandidate = indexDistancePairs.poll();
			while (neighbourCandidate.getIndex() == currentIndex)
				neighbourCandidate = indexDistancePairs.poll();
			neighbours[i] = neighbourCandidate.getIndex();
		}
		return neighbours;
	}

	/**
	 * Applies genetic operators and returns a new individual, based on the original
	 * provided.
	 *
	 * @param original
	 * @return new
	 */
	private Individual evolveNewIndividual(Individual original, int index, Random random) {
		// Check whether to apply mutation or crossover
		double r = random.nextDouble();
		int chosenOperation;
		if (crossoverProbability != 0.0 && r <= crossoverProbability)
			chosenOperation = CROSSOVER;
		else if (mutationProbability != 0.0 && r <= crossoverProbability + mutationProbability)
			chosenOperation = MUTATION;
		else if (localSearchProbability != 0.0 && r <= crossoverProbability + mutationProbability + localSearchProbability)
			chosenOperation = LOCAL_SEARCH;
		else
			throw new RuntimeException("Invalid random value when selecting ");

		// Perform crossover if that is the chosen operation
		if (chosenOperation == CROSSOVER) {
			// Select a neighbour at random
			int neighbourIndex = random.nextInt(numNeighbours);
			Individual neighbour = population[neighbourIndex];
			return crossOperator.doCrossover(original.clone(), neighbour.clone(), this);
		}
		// Else, perform mutation
		else if (chosenOperation == MUTATION) {
			return mutOperator.mutate(original.clone(), this);
		}
		// Else, perform local search
		else if (chosenOperation == LOCAL_SEARCH) {
			return localOperator.doSearch(original.clone(), this, index);
		}
		else {
			throw new RuntimeException("Invalid operation selected.");
		}
	}

	/**
	 * Updates the neighbours of the vector for a given index, changing their associated
	 * individuals to be the new individual (provided that this new individual is a better
	 * solution to the problem).
	 *
	 * @param newInd
	 * @param index
	 */
	private void updateNeighbours(Individual newInd, int index, Individual[] newGeneration) {
		// Retrieve neighbourhood indices
		int[] neighbourhoodIndices = neighbourhood[index];
		double newScore;
		if (tchebycheff)
			newScore = calculateTchebycheffScore(newInd, index);
		else
			newScore = calculateScore(newInd, index);
		double oldScore;
		for (int nIdx : neighbourhoodIndices) {
			// Calculate scores for the old solution, versus the new solution
			if (tchebycheff)
				oldScore = calculateTchebycheffScore(population[nIdx], index);
			else
				oldScore = calculateScore(population[nIdx], index);
			if (newScore < oldScore) {
				// Replace neighbour with new solution in new generation
				newGeneration[nIdx] = newInd.clone();
			}

		}
	}

	/**
	 * Calculates the problem score for a given individual, using a given
	 * set of weights.
	 *
	 * @param ind
	 * @param problemIndex - for retrieving weights
	 * @return score
	 */
	public double calculateScore(Individual ind, int problemIndex) {
		double[] problemWeights = weights[problemIndex];
		double sum = 0;
		for (int i = 0; i < numObjectives; i++)
			sum += (problemWeights[i]) * ind.getObjectiveValues()[i];
		return sum;
	}

	/**
	 * Calculates the problem score using the Tchebycheff approach, and the given
	 * set of weights.
	 *
	 * @param ind
	 * @param problemIndex - for retrieving weights and ideal point
	 * @return score
	 */
	public double calculateTchebycheffScore(Individual ind, int problemIndex) {
		double[] problemWeights = weights[problemIndex];
		double max_fun = -1 * Double.MAX_VALUE;

		for (int i = 0; i < numObjectives; i++) {
			double diff = abs(ind.getObjectiveValues()[i] - idealPoint[i]);
			double feval;
			if (problemWeights[i] == 0)
				feval = 0.00001 * diff;
			else
				feval = problemWeights[i] * diff;
			if (feval > max_fun)
				max_fun = feval;
		}
		return max_fun;
	}

	/**
	 * Updates the reference points if necessary, using the objective values
	 * of the individual provided.
	 *
	 * @param ind
	 */
	protected void updateReference(Individual ind) {
		for (int i = 0; i < numObjectives; i++) {
			if (ind.getObjectiveValues()[i] < idealPoint[i])
				idealPoint[i] = ind.getObjectiveValues()[i];
		}
	}

	/**
	 * Sorts the current population in order to identify the non-dominated
	 * solutions (i.e. the Pareto front).
	 *
	 * @param population
	 * @return Pareto front
	 */
	private Set<Individual> produceParetoFront(Set<Individual> population) {
		// Initialise sets/variable for tracking current front
		Set<Individual> front = new HashSet<Individual>();
		Set<Individual> toRemove = new HashSet<Individual>();
		boolean dominated = false;

		for (Individual i: population) {
			// Reset sets/variable
			toRemove.clear();
			dominated = false;

			/* Go through front and check whether the current individual should be added to it. Also
			 * check whether any individuals currently in the front should be removed (i.e. whether
			 * they are dominated by the current individual).*/
			for (Individual f: front) {
				if (i.dominates(f)) {
					toRemove.add(f);
				}
				else if (f.dominates(i)) {
					dominated = true;
				}
			}
			// Remove all dominated points from the Pareto set, and add current individual if not dominated
			front.removeAll(toRemove);
			if(!dominated)
				front.add(i);
		}
		return front;
	}

	/**
	 * Saves program statistics to the disk. This method should be called once per generation.
	 *
	 * @param writer - File writer for output.
	 * @param generation - current generation number.
	 */
	private void writeOutStatistics(FileWriter writer, int generation) {
		try {
			for (int i = 0; i < population.length; i++) {
				// Generation
				writer.append(String.format("%d ", generation));
				// Individual ID
				writer.append(String.format("%d ", i));
				// Breeding time
				writer.append(String.format("%d ", breedingTime[generation]));
				// Evaluation time
				writer.append(String.format("%d ", evaluationTime[generation]));
				// Rank and sparsity
				writer.append("0 0 ");
				// Objective one
				writer.append(String.format("%.20f ", population[i].getObjectiveValues()[0]));
				// Objective two
				writer.append(String.format("%.20f ", population[i].getObjectiveValues()[1]));
				// Raw availability
				writer.append(String.format("%.30f ", population[i].getAvailability()));
				// Raw reliability
				writer.append(String.format("%.30f ", population[i].getReliability()));
				// Raw time
				writer.append(String.format("%f ", population[i].getTime()));
				// Raw cost
				writer.append(String.format("%f\n", population[i].getCost()));
			}
		}
		catch (IOException e) {
			System.err.printf("Could not write to '%'.\n", outFileName);
			e.printStackTrace();
		}
	}

	/**
	 * Saves the Pareto front to the disk. This method should be called at the end of the run.
	 *
	 * @param writer - File writer for front.
	 * @param front - The set of individuals in the front.
	 */
	private void writeFrontStatistics(FileWriter writer, Set<Individual> front) {
		try {
			for (Individual ind : front) {
				// Rank and sparsity
				writer.append("0 0 ");
				// Objective one
				writer.append(String.format("%.20f ", ind.getObjectiveValues()[0]));
				// Objective two
				writer.append(String.format("%.20f ", ind.getObjectiveValues()[1]));
				// Raw availability
				writer.append(String.format("%.30f ", ind.getAvailability()));
				// Raw reliability
				writer.append(String.format("%.30f ", ind.getReliability()));
				// Raw time
				writer.append(String.format("%f ", ind.getTime()));
				// Raw cost
				writer.append(String.format("%f ", ind.getCost()));
				// Candidate string
				writer.append(String.format("\"%s\"\n", ind.toString()));
			}
		}
		catch (IOException e) {
			System.err.printf("Could not write to '%'.\n", outFileName);
			e.printStackTrace();
		}
	}

	/**
	 * Parses the WSC Web service file with the given name, creating Web
	 * services based on this information and saving them to the service map.
	 *
	 * @param fileName
	 */
	private void parseWSCServiceFile(String fileName) {
        Set<String> inputs = new HashSet<String>();
        Set<String> outputs = new HashSet<String>();
        double[] qos = new double[4];

        try {
        	File fXmlFile = new File(fileName);
        	DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
        	DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
        	Document doc = dBuilder.parse(fXmlFile);

        	NodeList nList = doc.getElementsByTagName("service");

        	for (int i = 0; i < nList.getLength(); i++) {
        		org.w3c.dom.Node nNode = nList.item(i);
        		Element eElement = (Element) nNode;

        		String name = eElement.getAttribute("name");
    		    qos[TIME] = Double.valueOf(eElement.getAttribute("Res"));
    		    qos[COST] = Double.valueOf(eElement.getAttribute("Pri"));
    		    qos[AVAILABILITY] = Double.valueOf(eElement.getAttribute("Ava"));
    		    qos[RELIABILITY] = Double.valueOf(eElement.getAttribute("Rel"));

				// Get inputs
				org.w3c.dom.Node inputNode = eElement.getElementsByTagName("inputs").item(0);
				NodeList inputNodes = ((Element)inputNode).getElementsByTagName("instance");
				for (int j = 0; j < inputNodes.getLength(); j++) {
					org.w3c.dom.Node in = inputNodes.item(j);
					Element e = (Element) in;
					inputs.add(e.getAttribute("name"));
				}

				// Get outputs
				org.w3c.dom.Node outputNode = eElement.getElementsByTagName("outputs").item(0);
				NodeList outputNodes = ((Element)outputNode).getElementsByTagName("instance");
				for (int j = 0; j < outputNodes.getLength(); j++) {
					org.w3c.dom.Node out = outputNodes.item(j);
					Element e = (Element) out;
					outputs.add(e.getAttribute("name"));
				}

                Service ws = new Service(name, qos, inputs, outputs);
                serviceMap.put(name, new Node(ws));
                inputs = new HashSet<String>();
                outputs = new HashSet<String>();
                qos = new double[4];
        	}
        }
        catch(IOException ioe) {
            System.out.println("Service file parsing failed...");
        }
        catch (ParserConfigurationException e) {
            System.out.println("Service file parsing failed...");
		}
        catch (SAXException e) {
            System.out.println("Service file parsing failed...");
		}
    }

	/**
	 * Parses the WSC task file with the given name, extracting input and
	 * output values to be used as the composition task.
	 *
	 * @param fileName
	 */
	private void parseWSCTaskFile(String fileName) {
		try {
	    	File fXmlFile = new File(fileName);
	    	DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
	    	DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
	    	Document doc = dBuilder.parse(fXmlFile);

	    	org.w3c.dom.Node provided = doc.getElementsByTagName("provided").item(0);
	    	NodeList providedList = ((Element) provided).getElementsByTagName("instance");
	    	taskInput = new HashSet<String>();
	    	for (int i = 0; i < providedList.getLength(); i++) {
				org.w3c.dom.Node item = providedList.item(i);
				Element e = (Element) item;
				taskInput.add(e.getAttribute("name"));
	    	}

	    	org.w3c.dom.Node wanted = doc.getElementsByTagName("wanted").item(0);
	    	NodeList wantedList = ((Element) wanted).getElementsByTagName("instance");
	    	taskOutput = new HashSet<String>();
	    	for (int i = 0; i < wantedList.getLength(); i++) {
				org.w3c.dom.Node item = wantedList.item(i);
				Element e = (Element) item;
				taskOutput.add(e.getAttribute("name"));
	    	}
		}
		catch (ParserConfigurationException e) {
            System.out.println("Task file parsing failed...");
            e.printStackTrace();
		}
		catch (SAXException e) {
            System.out.println("Task file parsing failed...");
            e.printStackTrace();
		}
		catch (IOException e) {
            System.out.println("Task file parsing failed...");
            e.printStackTrace();
		}
	}

	/**
	 * Parses the WSC taxonomy file with the given name, building a
	 * tree-like structure.
	 *
	 * @param fileName
	 */
	private void parseWSCTaxonomyFile(String fileName) {
		try {
	    	File fXmlFile = new File(fileName);
	    	DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
	    	DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
	    	Document doc = dBuilder.parse(fXmlFile);
	    	NodeList taxonomyRoots = doc.getChildNodes();

	    	processTaxonomyChildren(null, taxonomyRoots);
		}

		catch (ParserConfigurationException e) {
            System.err.println("Taxonomy file parsing failed...");
		}
		catch (SAXException e) {
            System.err.println("Taxonomy file parsing failed...");
		}
		catch (IOException e) {
            System.err.println("Taxonomy file parsing failed...");
		}
	}

	/**
	 * Recursive function for recreating taxonomy structure from file.
	 *
	 * @param parent - Nodes' parent
	 * @param nodes
	 */
	private void processTaxonomyChildren(TaxonomyNode parent, NodeList nodes) {
		if (nodes != null && nodes.getLength() != 0) {
			for (int i = 0; i < nodes.getLength(); i++) {
				org.w3c.dom.Node ch = nodes.item(i);

				if (!(ch instanceof Text)) {
					Element currNode = (Element) nodes.item(i);
					String value = currNode.getAttribute("name");
					TaxonomyNode taxNode = taxonomyMap.get( value );
					if (taxNode == null) {
					    taxNode = new TaxonomyNode(value);
					    taxonomyMap.put( value, taxNode );
					}
					if (parent != null) {
					    taxNode.parents.add(parent);
						parent.children.add(taxNode);
					}

					NodeList children = currNode.getChildNodes();
					processTaxonomyChildren(taxNode, children);
				}
			}
		}
	}

	/**
	 * Converts input, output, and service instance values to their corresponding
	 * ontological parent.
	 */
	private void findConceptsForInstances() {
		Set<String> temp = new HashSet<String>();

		for (String s : taskInput)
			temp.add(taxonomyMap.get(s).parents.get(0).value);
		taskInput.clear();
		taskInput.addAll(temp);

		temp.clear();
		for (String s : taskOutput)
				temp.add(taxonomyMap.get(s).parents.get(0).value);
		taskOutput.clear();
		taskOutput.addAll(temp);

		for (Node s : serviceMap.values()) {
			temp.clear();
			Set<String> inputs = s.getInputs();
			for (String i : inputs)
				temp.add(taxonomyMap.get(i).parents.get(0).value);
			inputs.clear();
			inputs.addAll(temp);

			temp.clear();
			Set<String> outputs = s.getOutputs();
			for (String o : outputs)
				temp.add(taxonomyMap.get(o).parents.get(0).value);
			outputs.clear();
			outputs.addAll(temp);
		}
	}

	/**
	 * Populates the taxonomy tree by associating services to the
	 * nodes in the tree.
	 */
	private void populateTaxonomyTree() {
		for (Node s: serviceMap.values()) {
			addServiceToTaxonomyTree(s);
		}
	}

	private void addServiceToTaxonomyTree(Node s) {
		// Populate outputs
	    Set<TaxonomyNode> seenConceptsOutput = new HashSet<TaxonomyNode>();
		for (String outputVal : s.getOutputs()) {
			TaxonomyNode n = taxonomyMap.get(outputVal);
			s.getTaxonomyOutputs().add(n);

			// Also add output to all parent nodes
			Queue<TaxonomyNode> queue = new LinkedList<TaxonomyNode>();
			queue.add( n );

			while (!queue.isEmpty()) {
			    TaxonomyNode current = queue.poll();
		        seenConceptsOutput.add( current );
		        current.servicesWithOutput.add(s);
		        for (TaxonomyNode parent : current.parents) {
		            if (!seenConceptsOutput.contains( parent )) {
		                queue.add(parent);
		                seenConceptsOutput.add(parent);
		            }
		        }
			}
		}
		// Populate inputs
		Set<TaxonomyNode> seenConceptsInput = new HashSet<TaxonomyNode>();
		for (String inputVal : s.getInputs()) {
			TaxonomyNode n = taxonomyMap.get(inputVal);

			// Also add input to all children nodes
			Queue<TaxonomyNode> queue = new LinkedList<TaxonomyNode>();
			queue.add( n );

			while(!queue.isEmpty()) {
				TaxonomyNode current = queue.poll();
				seenConceptsInput.add( current );

			    Set<String> inputs = current.servicesWithInput.get(s);
			    if (inputs == null) {
			    	inputs = new HashSet<String>();
			    	inputs.add(inputVal);
			    	current.servicesWithInput.put(s, inputs);
			    }
			    else {
			    	inputs.add(inputVal);
			    }

			    for (TaxonomyNode child : current.children) {
			        if (!seenConceptsInput.contains( child )) {
			            queue.add(child);
			            seenConceptsInput.add( child );
			        }
			    }
			}
		}
		return;
	}

	/**
	 * Goes through the service list and retrieves only those services which
	 * could be part of the composition task requested by the user.
	 *
	 * @param serviceMap
	 * @return relevant services
	 */
	private Set<Node> getRelevantServices(Map<String,Node> serviceMap, Set<String> inputs, Set<String> outputs) {
		// Copy service map values to retain original
		Collection<Node> services = new ArrayList<Node>(serviceMap.values());

		Set<String> cSearch = new HashSet<String>(inputs);
		Set<Node> sSet = new HashSet<Node>();
		int layer = 0;
		Set<Node> sFound = discoverService(services, cSearch);
		while (!sFound.isEmpty()) {
			sSet.addAll(sFound);
			// Record the layer that the services belong to in each node
			for (Node s : sFound)
				s.setLayer(layer);

			layer++;
			services.removeAll(sFound);
			for (Node s: sFound) {
				cSearch.addAll(s.getOutputs());
			}
			sFound.clear();
			sFound = discoverService(services, cSearch);
		}

		numLayers = layer;

		if (isSubsumed(outputs, cSearch)) {
			return sSet;
		}
		else {
			String message = "It is impossible to perform a composition using the services and settings provided.";
			System.out.println(message);
			System.exit(0);
			return null;
		}
	}

	/**
	 * Discovers all services from the provided collection whose
	 * input can be satisfied either (a) by the input provided in
	 * searchSet or (b) by the output of services whose input is
	 * satisfied by searchSet (or a combination of (a) and (b)).
	 *
	 * @param services
	 * @param searchSet
	 * @return set of discovered services
	 */
	private Set<Node> discoverService(Collection<Node> services, Set<String> searchSet) {
		Set<Node> found = new HashSet<Node>();
		for (Node s: services) {
			if (isSubsumed(s.getInputs(), searchSet))
				found.add(s);
		}
		return found;
	}

	/**
	 * Checks whether set of inputs can be completely satisfied by the search
	 * set, making sure to check descendants of input concepts for the subsumption.
	 *
	 * @param inputs
	 * @param searchSet
	 * @return true if search set subsumed by input set, false otherwise.
	 */
	public boolean isSubsumed(Set<String> inputs, Set<String> searchSet) {
		boolean satisfied = true;
		for (String input : inputs) {
			Set<String> subsumed = taxonomyMap.get(input).getSubsumedConcepts();
			if (!isIntersection( searchSet, subsumed )) {
				satisfied = false;
				break;
			}
		}
		return satisfied;
	}

    private static boolean isIntersection( Set<String> a, Set<String> b ) {
        for ( String v1 : a ) {
            if ( b.contains( v1 ) ) {
                return true;
            }
        }
        return false;
    }

	private void calculateNormalisationBounds(Set<Node> services) {
		for(Node service: services) {
			double[] qos = service.getQos();

			// Availability
			double availability = qos[AVAILABILITY];
			if (availability > maxAvailability)
				maxAvailability = availability;

			// Reliability
			double reliability = qos[RELIABILITY];
			if (reliability > maxReliability)
				maxReliability = reliability;

			// Time
			double time = qos[TIME];
			if (time > maxTime)
				maxTime = time;
			if (time < minTime)
				minTime = time;

			// Cost
			double cost = qos[COST];
			if (cost > maxCost)
				maxCost = cost;
			if (cost < minCost)
				minCost = cost;
		}
		// Adjust max. cost and max. time based on the number of services in shrunk repository
		maxCost *= services.size();
		maxTime *= services.size();

	}

	/**
	 * This method finishes calculating the objective values for each individual
	 * according to the QoS bounds found for this generation. If using dynamic
	 * normalisation, bounds are updated in this process.
	 *
	 * @param state
	 * @param threadnum
	 */
	public void finishEvaluating() {
		// Get population
		double minA = 2.0;
		double maxA = -1.0;
		double minR = 2.0;
		double maxR = -1.0;
		double minT = Double.MAX_VALUE;
		double maxT = -1.0;
		double minC = Double.MAX_VALUE;
		double maxC = -1.0;

		// Find the normalisation bounds
		for (Individual ind : population) {
			double a = ind.getAvailability();
			double r = ind.getReliability();
			double t = ind.getTime();
			double c = ind.getCost();

			if (dynamicNormalisation) {
				if (a < minA)
					minA = a;
				if (a > maxA)
					maxA = a;
				if (r < minR)
					minR = r;
				if (r > maxR)
					maxR = r;
				if (t < minT)
					minT = t;
				if (t > maxT)
					maxT = t;
				if (c < minC)
					minC = c;
				if (c > maxC)
					maxC = c;
			}
		}

		if (dynamicNormalisation) {
			// Update the normalisation bounds with the newly found values
			minAvailability = minA;
			maxAvailability = maxA;
			minReliability = minR;
			maxReliability = maxR;
			minCost = minC;
			maxCost = maxC;
			minTime = minT;
			maxTime = maxT;

			// Finish calculating the fitness of each candidate
			for (Individual ind : population) {
				ind.finishCalculatingFitness();
			}
		}
	}

	public GraphIndividual createNewGraph(GraphIndividual mergedGraph, Node start, Node end, Set<Node> relevant) {
		GraphIndividual newGraph = new GraphIndividual();

		Set<String> currentEndInputs = new HashSet<String>();
		Map<String,Edge> connections = new HashMap<String,Edge>();

		// Connect start node
		connectCandidateToGraphByInputs(start, connections, newGraph, currentEndInputs);

		Set<Node> seenNodes = new HashSet<Node>();
		List<Node> candidateList = new ArrayList<Node>();

		if (mergedGraph != null)
			addToCandidateListFromEdges(start, mergedGraph, seenNodes, candidateList);
		else
			addToCandidateList(start, seenNodes, relevant, candidateList);

		Collections.shuffle(candidateList, random);

		finishConstructingGraph(currentEndInputs, end, candidateList, connections, newGraph, mergedGraph, seenNodes, relevant, true);

		newGraph.setInit(this);
		return newGraph;
	}

	public void addToCandidateList(Node n, Set<Node> seenNode, Set<Node> relevant, List<Node> candidateList) {
		seenNode.add(n);
		List<TaxonomyNode> taxonomyOutputs;
		if (n.getName().equals("start")) {
			taxonomyOutputs = new ArrayList<TaxonomyNode>();
			for (String outputVal : n.getOutputs()) {
				taxonomyOutputs.add(taxonomyMap.get(outputVal));
			}
		}
		else
			taxonomyOutputs = serviceMap.get(n.getName()).getTaxonomyOutputs();

		for (TaxonomyNode t : taxonomyOutputs) {
			// Add servicesWithInput from taxonomy node as potential candidates to be connected
			for (Node current : t.servicesWithInput.keySet()) {
				if (!seenNode.contains(current) && relevant.contains(current)) {
					candidateList.add(current);
					seenNode.add(current);
				}
			}
		}
	}

	private void addToCandidateListFromEdges (Node n, GraphIndividual mergedGraph, Set<Node> seenNode, List<Node> candidateList) {
		seenNode.add(n);

		Node original = mergedGraph.nodeMap.get(n.getName());

		for (Edge e : original.getOutgoingEdgeList()) {
			// Add servicesWithInput from taxonomy node as potential candidates to be connected
			Node current = e.getToNode();
			if (!seenNode.contains(current)) {
				candidateList.add(current);
				seenNode.add(current);
			}
		}
	}

	public void finishConstructingGraph(Set<String> currentEndInputs, Node end, List<Node> candidateList, Map<String,Edge> connections, GraphIndividual newGraph, GraphIndividual mergedGraph, Set<Node> seenNodes, Set<Node> relevant, boolean shuffleCandidates) {

	 // While end cannot be connected to graph
		while(!checkCandidateNodeSatisfied(connections, newGraph, end, end.getInputs(), null)){
			connections.clear();

            // Select node
            int index;

            candidateLoop:
            for (index = 0; index < candidateList.size(); index++) {
                Node candidate = candidateList.get(index).clone();
                // For all of the candidate inputs, check that there is a service already in the graph
                // that can satisfy it

                if (!checkCandidateNodeSatisfied(connections, newGraph, candidate, candidate.getInputs(), null)) {
                    connections.clear();
                	continue candidateLoop;
                }

                // Connect candidate to graph, adding its reachable services to the candidate list
                connectCandidateToGraphByInputs(candidate, connections, newGraph, currentEndInputs);
                connections.clear();

                if (mergedGraph != null)
                    addToCandidateListFromEdges(candidate, mergedGraph, seenNodes, candidateList);
                else
                    addToCandidateList(candidate, seenNodes, relevant, candidateList);

                break;
            }

            candidateList.remove(index);
            if (shuffleCandidates)
            	Collections.shuffle(candidateList, random);
        }

        connectCandidateToGraphByInputs(end, connections, newGraph, currentEndInputs);
        connections.clear();
        removeDanglingNodes(newGraph);
	}

	public void connectCandidateToGraphByInputs(Node candidate, Map<String,Edge> connections, GraphIndividual graph, Set<String> currentEndInputs) {

		graph.nodeMap.put(candidate.getName(), candidate);
		graph.edgeList.addAll(connections.values());
		candidate.getIncomingEdgeList().addAll(connections.values());

		for (Edge e : connections.values()) {
			Node fromNode = graph.nodeMap.get(e.getFromNode().getName());
			fromNode.getOutgoingEdgeList().add(e);
		}
		for (String o : candidate.getOutputs()) {
			currentEndInputs.addAll(taxonomyMap.get(o).endNodeInputs);
		}
	}

	private boolean checkCandidateNodeSatisfied(Map<String, Edge> connections, GraphIndividual newGraph, Node candidate, Set<String> candInputs, Set<Node> fromNodes) {

		Set<String> candidateInputs = new HashSet<String>(candInputs);
		Set<String> startIntersect = new HashSet<String>();

		// Check if the start node should be considered
		Node start = newGraph.nodeMap.get("start");

		if (fromNodes == null || fromNodes.contains(start)) {
    		for(String output : start.getOutputs()) {
    			Set<String> inputVals = taxonomyMap.get(output).servicesWithInput.get(candidate);
    			if (inputVals != null) {
    				candidateInputs.removeAll(inputVals);
    				startIntersect.addAll(inputVals);
    			}
    		}

    		if (!startIntersect.isEmpty()) {
    			Edge startEdge = new Edge(startIntersect);
    			startEdge.setFromNode(start);
    			startEdge.setToNode(candidate);
    			connections.put(start.getName(), startEdge);
    		}
		}


		for (String input : candidateInputs) {
			boolean found = false;
			for (Node s : taxonomyMap.get(input).servicesWithOutput) {
			    if (fromNodes == null || fromNodes.contains(s)) {
    				if (newGraph.nodeMap.containsKey(s.getName())) {
    					Set<String> intersect = new HashSet<String>();
    					intersect.add(input);

    					Edge mapEdge = connections.get(s.getName());
    					if (mapEdge == null) {
    						Edge e = new Edge(intersect);
    						e.setFromNode(newGraph.nodeMap.get(s.getName()));
    						e.setToNode(candidate);
    						connections.put(e.getFromNode().getName(), e);
    					} else
    						mapEdge.getIntersect().addAll(intersect);

    					found = true;
    					break;
    				}
			    }
			}
			// If that input cannot be satisfied, move on to another candidate
			// node to connect
			if (!found) {
				// Move on to another candidate
				return false;
			}
		}
		return true;
	}

	public void removeDanglingNodes(GraphIndividual graph) {
	    List<Node> dangling = new ArrayList<Node>();
	    for (Node g : graph.nodeMap.values()) {
	        if (!g.getName().equals("end") && g.getOutgoingEdgeList().isEmpty()) {
	            dangling.add( g );
	        }
	    }

	    for (Node d: dangling) {
	        removeDangling(d, graph);
	    }
	}

	private void removeDangling(Node n, GraphIndividual graph) {
	    if (n.getOutgoingEdgeList().isEmpty()) {
	        graph.nodeMap.remove( n.getName() );
	        for (Edge e : n.getIncomingEdgeList()) {
	            e.getFromNode().getOutgoingEdgeList().remove( e );
	            graph.edgeList.remove( e );
	            removeDangling(e.getFromNode(), graph);
	        }
	    }
	}

	/**
	 * Application entry point.
	 *
	 * @param args
	 */
	public static void main(String[] args) {
		if (args.length == 0)
			throw new RuntimeException("A parameters file should be provided as an argument.");
		new MOEAD(args);
	}
}
