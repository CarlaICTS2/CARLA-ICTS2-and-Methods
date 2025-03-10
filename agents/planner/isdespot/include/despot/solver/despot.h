#ifndef DESPOT_H
#define DESPOT_H

#include <despot/core/solver.h>
#include <despot/interface/pomdp.h>
#include <despot/interface/belief.h>
#include <despot/core/node.h>
#include <despot/core/globals.h>
#include <despot/core/history.h>
#include <despot/random_streams.h>
#include <despot/util/util.h>

class TCPConnector;

namespace despot {

class DESPOT: public Solver {
friend class VNode;

protected:
	VNode* root_;

	ScenarioLowerBound* lower_bound_;
	ScenarioUpperBound* upper_bound_;

public:
	// for HyLEAP
	std::vector<double> action_probabilities;

	static TCPConnector* eval_conn;
	static std::vector<double> previous_root_lstm_state;
	static ValuedAction previous_action;

	DESPOT(const DSPOMDP* model, ScenarioLowerBound* lb, ScenarioUpperBound* ub, Belief* belief = NULL);
	DESPOT(const DSPOMDP* model, ScenarioLowerBound* lb, ScenarioUpperBound* ub, TCPConnector* eval_conn, Belief* belief = NULL);
	virtual ~DESPOT();

	ValuedAction Search();

	static VNode* ConstructTree(
		std::vector<State*>& particles, 
		RandomStreams& streams,
		ScenarioLowerBound* lower_bound, 
		ScenarioUpperBound* upper_bound,
		const DSPOMDP* model, 
		History& history, 
		double timeout
	);

	static VNode* HackyConstructTree(std::vector<State*>& particles, 
									 RandomStreams& streams,
									 ScenarioLowerBound* lower_bound, 
									 ScenarioUpperBound* upper_bound,
									 const DSPOMDP* model, 
									 History& history, 
									 double timeout);

protected:
	static VNode* Trial(VNode* root, 
						RandomStreams& streams,
						ScenarioLowerBound* lower_bound, 
						ScenarioUpperBound* upper_bound,
						const DSPOMDP* model, 
						History& history);
	static VNode* HackyTrial(VNode* root, 
							 RandomStreams& streams,
							 ScenarioLowerBound* lower_bound, 
							 ScenarioUpperBound* upper_bound,
							 const DSPOMDP* model, 
							 History& history);

	void compute_action_policy(VNode* node);

	static void network_evaluate(VNode* expanded_node);
	static void network_evaluate(std::vector<VNode*> expanded_nodes, bool root = false);

	static void InitLowerBound(VNode* vnode, 
							   ScenarioLowerBound* lower_bound,
							   RandomStreams& streams, 
							   History& history);
	static void InitUpperBound(VNode* vnode, 
							   ScenarioUpperBound* upper_bound,
							   RandomStreams& streams, 
							   History& history);
	static void InitBounds(VNode* vnode, 
						   ScenarioLowerBound* lower_bound,
						   ScenarioUpperBound* upper_bound, 
						   RandomStreams& streams, 
						   History& history);

	static void InitUpperBoundHyLEAP(VNode* vnode);
	static void InitBoundsHyLEAP(VNode* vnode, 
								 ScenarioLowerBound* lower_bound, 
								 RandomStreams& streams, 
								 History& history);

	static void Expand(VNode* vnode,
					   ScenarioLowerBound* lower_bound, 
					   ScenarioUpperBound* upper_bound,
					   const DSPOMDP* model, 
					   RandomStreams& streams, 
					   History& history);
	static void Expand(QNode* qnode, 
					   ScenarioLowerBound* lower_bound,
					   ScenarioUpperBound* upper_bound, 
					   const DSPOMDP* model,
					   RandomStreams& streams, 
					   History& history);

	static int ExpandHyLEAP(VNode* vnode,
						    ScenarioLowerBound* lower_bound, 
						    ScenarioUpperBound* upper_bound,
						    const DSPOMDP* model, 
						    RandomStreams& streams,
						    History& history);
	static int ExpandHyLEAP(QNode* qnode, 
							ScenarioLowerBound* lb,
							ScenarioUpperBound* ub, 
							const DSPOMDP* model,
							RandomStreams& streams,
							History& history,
							std::vector<VNode*>& expanded_nodes);

	static void Backup(VNode* vnode);
	static void Update(VNode* vnode);
	static void Update(QNode* qnode);

	static double gap(VNode* vnode);
	static double weu(VNode* vnode);
	static double weu(VNode* vnode, double epsilon);

	static VNode* select_best_weu_node(QNode* qnode);
	static QNode* select_best_upper_bound_node(VNode* vnode);
	static ValuedAction compute_optimal_action(VNode* vnode);

	ScenarioLowerBound* lower_bound() const;
	ScenarioUpperBound* upper_bound() const;

	static VNode* find_blocker(VNode* vnode);
	static void exploit_blocker(VNode* vnode);
};

} // namespace despot

#endif
