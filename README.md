# otclustering
Optimal Transport Tools developed in Conjunction with Thesis Research

OPTIMAL TRANSPORT CLUSTERING OF GEOMETRIC AND SPATIAL DATA

We study how an optimal assignment can be determined via the Kantorovich formulation of optimal transport, ensuring that voters are assigned to voting centers in a way that minimizes aggregate transport cost.
	
Early computational experiments, visualized through color-coded district assignments using tools in this repository, revealed unexpected non-contiguous segments---small patches of one color (voter assignment) embedded within regions dominated by a different assignment. We referred to these as ``sprinkles'', as they appeared as isolated clusters of assigned voters surrounded by different voting regions.
	
OT theory- particularly the characterization of optimal plans in the Euclidian case with square cost- suggests that partitions in an optimal configuration ought to be connected when the cost is a graph distance.
	
However, upon deeper investigation-including:
  graph-based OT models, where we analyzed voter assignments as flows on a network;
  validation via the Beckmann formulation, which recasts OT as a flow minimization problem; and
  experiments in 1-D, where we observed similar discontinuities in much simpler settings,

it became evident that these anomalies were not numerical errors, but rather fundamental consequences of using a discrete L1 cost function on the graph. Through further analysis, we formally characterized the conditions under which such sprinkling effects emerge using the original Python tools provided in this repository.
