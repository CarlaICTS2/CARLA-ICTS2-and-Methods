#ifndef NODE_H
#define NODE_H

#include <despot/interface/pomdp.h>
#include <despot/util/util.h>
#include <despot/random_streams.h>
#include <despot/util/logging.h>
#include "../../../carla_car/src/param.h"


namespace despot {

class QNode;

/* =============================================================================
 * VNode class
 * =============================================================================*/

/**
 * A belief/value/AND node in the search tree.
 */
class VNode 
{
	protected:
		// tree size of time of creation -1 (such that the root node is b0)
		int id_; 
		// particle set approximating the belief
		std::vector<State*> particles_; 
		// depth in the belief tree
		int depth_;
		// action that led to this observation
		QNode* parent_;
		// hash of this observation
		OBS_TYPE edge_;
		// children after having executed all actions
		std::vector<QNode*> children_;
		// value and action given by default policy
		ValuedAction default_move_;
		double lower_bound_;
		double upper_bound_;
		// number of visits on the node
		int count_; 

	public:
		VNode* vstar;
		double utility_upper_bound;

		// HyLEAP & HyPLAN, drla = deep reinforcement learning agent
		ValuedAction drla_prediction;
		std::vector<double> lstm_state;

		VNode(std::vector<State*>& particles, int id, int depth = 0, QNode* parent = NULL, OBS_TYPE edge = -1);
		~VNode();

		int id() const;

		const std::vector<State*>& particles() const;
		void depth(int d);
		int depth() const;

		void parent(QNode* parent);
		QNode* parent();
		OBS_TYPE edge();

		double Weight() const;

		const std::vector<QNode*>& children() const;
		std::vector<QNode*>& children();

		const QNode* Child(ACT_TYPE action) const;
		QNode* Child(ACT_TYPE action);
		int Size() const;
		int PolicyTreeSize() const;

		void default_move(ValuedAction move);
		ValuedAction default_move() const;

		void lower_bound(double value);
		double lower_bound() const;

		void upper_bound(double value);
		double upper_bound() const;

		int particle_num();

		bool IsLeaf();

		void count(int c);
		int count() const;

		void PrintTree(int depth = -1, std::ostream& os = std::cout);
		void PrintPolicyTree(int depth = -1, std::ostream& os = std::cout);

		void Free(const DSPOMDP& model);
};

/* =============================================================================
 * QNode class
 * =============================================================================*/

/**
 * A Q-node/AND-node (child of a belief node) of the search tree.
 */
class QNode 
{
	protected:
		VNode* parent_;
		ACT_TYPE edge_;
		std::map<OBS_TYPE, VNode*> children_;
		double lower_bound_;
		double upper_bound_;
		int count_; // Number of visits on the node

	public:
		double default_value;
		double utility_upper_bound;
		double step_reward;
		double likelihood;
		VNode* vstar;

		QNode(VNode* parent, int edge);
		~QNode();

		void parent(VNode* parent);
		VNode* parent();
		
		int edge();

		std::map<OBS_TYPE, VNode*>& children();
		VNode* Child(OBS_TYPE obs);

		int Size() const;
		int PolicyTreeSize() const;

		double Weight() const;

		void lower_bound(double value);
		double lower_bound() const;

		void upper_bound(double value);
		double upper_bound() const;

		void count(int c);
		int count() const;
};

} // namespace despot

#endif
