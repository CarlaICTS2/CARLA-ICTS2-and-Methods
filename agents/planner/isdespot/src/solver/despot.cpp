#include <despot/solver/despot.h>
#include "despot/util/logging.h"

#include "../../carla_car/src/connector.h"
#include "../../carla_car/src/state.h"


// ===========================================================================================
//					  TRACKING VARIABLES (for a single DESPOT construction)
// ===========================================================================================
// track planning effort metrics
static int tree_nodes = 0;
static int backups = 0;
static int terminal_states = 0;

static std::vector<double> root_lower_bounds;
static std::vector<double> root_upper_bounds;

static std::vector<double> initial_lower_bounds;
static std::vector<double> initial_upper_bounds;

static std::vector<double> trial_depths; 
static std::vector<double> excess_uncertainty;
static std::vector<double> observations;

// fined-grained execution time tracking (in ms)
static double search_time = 0;
static double construct_tree_time = 0;

static double trial_time = 0;
static double expand_time = 0;
static double qcreation_time = 0;
static double copy_time = 0;
static double step_time = 0;
static double free_time = 0;
static double norm_time = 0;
static double vcreation_time = 0;
static double init_bounds_time = 0;

static double backup_time = 0;

// hyleap 
static double python_communication_time = 0;
static int python_interactions = 0;

// hyplan
static std::vector<double> uncertainty_values;

using namespace std;

namespace despot {

TCPConnector* DESPOT::eval_conn = nullptr;
// initialize lstm state to be all 0 for the first step of any given episode
std::vector<double> DESPOT::previous_root_lstm_state(ModelParams::LSTM_STATE_SIZE, 0);
ValuedAction DESPOT::previous_action;

// ===========================================================================================
//											HYLEAP
// ===========================================================================================

// only called by the root node of each step
void DESPOT::network_evaluate(VNode* expanded_node) 
{
	logd << __FUNCTION__ << endl;
	// set the lstm state of the current root to the lstm state of the previous root node
	expanded_node->lstm_state = DESPOT::previous_root_lstm_state;
	std::vector<VNode*> expanded_nodes(1, expanded_node);
	network_evaluate(expanded_nodes, true);
	// save the new lstm state for the next step's root node
	DESPOT::previous_root_lstm_state = expanded_node->lstm_state;
}


void DESPOT::network_evaluate(std::vector<VNode*> expanded_nodes, bool root)
{
	const auto enter = Clock::now();

    eval_conn->send_expanded_nodes(expanded_nodes, root);
    eval_conn->receive_drla_predictions(expanded_nodes);

	python_communication_time += (std::chrono::duration_cast<dsec>(Clock::now() - enter).count() * 1000);
	python_interactions += expanded_nodes.size();
}


// essentially assigns a probability of 1 to the action with the highest lower bound
void DESPOT::compute_action_policy(VNode* node)
{
	logd << __FUNCTION__ << endl;
    int num_children = node->children().size();

    DESPOT::action_probabilities = {0, 0, 0};

    if(num_children > 3){
        std::cerr << "Invalid number of root children: Expected 3, got " << num_children << endl;
        exit(-1);
    }

	// root has no children
	if (num_children < 3) {
		logd << "\t- root has no children: only the default move is available (determined by the reactive controller" << endl;
		// only the default move is available (determined by the reactive controller)
		action_probabilities[node->default_move().action] = 1;
		// all other actions have probability of 0
		return;
	}

	double values[3];
    double exp_values[3];
    double sum = 0;
    double maxVal = Globals::NEG_INFTY;
    double tau = 0.1;
    int actionIndex = 0;

    for (int action = 0; action < num_children; action++) {
        const QNode* qnode = node->Child(action);
        double val = qnode->lower_bound();
        if(val < -2){
            val = -2;
        }
        if(val > 2){
            val = 2;
        }
        val = val / tau;
        if(val > maxVal){
            maxVal = val;
            actionIndex = action;
        }
        values[action] = val;
    }

    for (int action = 0; action < num_children; action++) {
        exp_values[action] = exp((values[action] - maxVal));
        sum += exp_values[action];
    }

    for (int action = 0; action < num_children; action++) {
        action_probabilities[action] = exp_values[action] / sum;
		logd << "\t- action: " << ModelParams::action_idx_to_string[action]  
			 << " with l(b): " << node->Child(action)->lower_bound()
			 << " and p(action): " << action_probabilities[action] << endl;
    }
}

// ===========================================================================================
//											IS-DESPOT
// ===========================================================================================
DESPOT::DESPOT(const DSPOMDP* model, 
			   ScenarioLowerBound* lb, 
			   ScenarioUpperBound* ub, 
			   Belief* belief) :
			   Solver(model, belief),
			   root_(NULL), 
			   lower_bound_(lb),
			   upper_bound_(ub) 
{
	if (model == NULL) {
		printf("IS-DESPOT::[%s] Invalid POMDP model: NULL.\n", __PRETTY_FUNCTION__);
		exit(-1);
	}
}


DESPOT::DESPOT(const DSPOMDP* model, 
			   ScenarioLowerBound* lb, 
			   ScenarioUpperBound* ub, 
			   TCPConnector* eval_conn, 
			   Belief* belief) :
			   Solver(model, belief),
			   root_(NULL),
			   lower_bound_(lb),
			   upper_bound_(ub)
{
	if (model == NULL) {
		printf("IS-DESPOT::[%s] Invalid POMDP model: NULL.\n", __PRETTY_FUNCTION__);
		exit(-1);		
	}
	this->eval_conn = eval_conn;
	if (this->eval_conn == nullptr) {
		printf("IS-DESPOT::[%s] Invalid NN evaluation connection: nullptr.\n", __PRETTY_FUNCTION__);
		exit(-1);
	}
}


DESPOT::~DESPOT() 
{
	if(lower_bound_!=NULL) {
		delete lower_bound_;
		lower_bound_ = NULL;
	}
	if(upper_bound_!=NULL) {
		delete upper_bound_;
		upper_bound_ = NULL;
	}
	if(model_!=NULL) {
		delete model_;
		model_ = NULL;
	}
}


ValuedAction DESPOT::Search() 
{
	logd << __FUNCTION__ << endl;

	const auto enter = Clock::now();
	// reset tracking variables for each construction
	tree_nodes = 0, backups = 0, terminal_states = 0;
	
	initial_lower_bounds.clear();
	initial_upper_bounds.clear();

	root_lower_bounds.clear();
	root_upper_bounds.clear();

	trial_depths.clear();
	excess_uncertainty.clear();
	observations.clear();

	uncertainty_values.clear();

	python_communication_time = 0, python_interactions = 0;

	// execution time tracking
	search_time = 0, construct_tree_time = 0, trial_time = 0, expand_time = 0, qcreation_time = 0, copy_time = 0;
	step_time = 0, free_time = 0, norm_time = 0, vcreation_time = 0, init_bounds_time = 0, backup_time = 0;

	// return a random action if no time is allocated for planning
	if (Globals::config.TIMEOUT <= 0) {
		return ValuedAction(Random::RANDOM.NextInt(model_->NumActions()), Globals::NEG_INFTY);
	}

	logb << "\t- belief before resampling: " << endl
		 << belief_->print() << endl;
	// sample states
	vector<State*> particles = belief_->Sample(Globals::config.PARTICLE_NUMBER);
	logb << "\t- belief after resampling: " << endl;
	model_->PrintParticles(particles);

	// construct scenarios (stream of random numbers + initially sampled state)
	static RandomStreams streams = RandomStreams(Globals::config.PARTICLE_NUMBER, Globals::config.MAX_SEARCH_DEPTH);
	lower_bound_->Init(streams);
	upper_bound_->Init(streams);

	// use Florian's hacky IS-DESPOT implementation (only available for HyLEAP and HyPLAN)
	if (Globals::config.HACKY) {
		root_ = HackyConstructTree(
			particles, streams, lower_bound_, upper_bound_, model_, history_, Globals::config.TIMEOUT
		);
	}
	// use "correct" IS-DESPOT implementation
	else {
		root_ = ConstructTree(
			particles, streams, lower_bound_, upper_bound_, model_, history_, Globals::config.TIMEOUT
		);
	}

	// make IS-DESPOT's action policy a valid probability distribution scaled according to temperature tau = 0.1
    compute_action_policy(root_);
	ValuedAction astar = compute_optimal_action(root_);

	// destroy pointers
	root_->Free(*model_);

	// exit search
	search_time = std::chrono::duration_cast<dsec>(Clock::now() - enter).count() * 1000;

	// remember step
	if (Globals::config.TRACKING) {
		if (tree_nodes != root_->Size()) {
			printf("IS-DESPOT::[%s] Incongruent tree nodes: Expected %d, got %d.\n",
			__PRETTY_FUNCTION__, root_->Size(), tree_nodes);
			exit(-1);
		}
		// sanity check: captured all tree nodes
		if (tree_nodes != initial_upper_bounds.size()) {
			printf("IS-DESPOT::[%s] Incongruent bound values: Expected %d, got %lu.\n",
			__PRETTY_FUNCTION__, tree_nodes, initial_upper_bounds.size());
			exit(-1);
		}

		// track palanning effort metrics
		logging::tree_nodes.push_back(tree_nodes);
		logging::backups.push_back(backups);
		logging::terminal_states.push_back(terminal_states);

		logging::root_lower_bounds.push_back(root_lower_bounds);
		logging::root_upper_bounds.push_back(root_upper_bounds);

		logging::initial_lower_bounds.push_back(initial_lower_bounds);
		logging::initial_upper_bounds.push_back(initial_upper_bounds);

		logging::trial_depths.push_back(trial_depths);
		logging::excess_uncertainties.push_back(excess_uncertainty);
		logging::observations.push_back(observations);

		// fine-grained execution time tracking
		logging::search_execution_times.push_back(search_time);
		logging::construct_tree_execution_times.push_back(construct_tree_time);
		logging::trial_execution_times.push_back(trial_time);
		logging::expand_execution_times.push_back(expand_time);
		logging::qcreation_execution_times.push_back(qcreation_time);
		logging::copy_execution_times.push_back(copy_time);
		logging::step_execution_times.push_back(step_time);
		logging::free_execution_times.push_back(free_time);
		logging::norm_execution_times.push_back(norm_time);
		logging::vcreation_execution_times.push_back(vcreation_time);
		logging::init_bounds_execution_times.push_back(init_bounds_time);

		logging::backup_execution_times.push_back(backup_time);

		if (Globals::config.AGENT == HyLEAP || Globals::config.AGENT == HyPLAN) {
			logging::python_communication_times.push_back(python_communication_time);
			logging::python_interactions.push_back(python_interactions);
		}

		if (Globals::config.AGENT == HyPLAN && !Globals::config.NO_VERTICAL_PRUNING) {
			logging::uncertainty_values.push_back(uncertainty_values);
		}
	}

	delete root_;

	return astar;
}


VNode* DESPOT::ConstructTree(vector<State*>& particles,
							 RandomStreams& streams,
							 ScenarioLowerBound* lower_bound, 
							 ScenarioUpperBound* upper_bound,
							 const DSPOMDP* model, 
							 History& history, 
							 double timeout) 
{
	logd << __FUNCTION__ << endl;
	const auto enter = Clock::now();

	for (int i = 0; i < particles.size(); i++)
		particles[i]->scenario_id = i;

	VNode* root = new VNode(particles, 0);
	tree_nodes += 1;

	// only hyleap and hyplan require a special bound initialization
	// "hacky" will never reach here, because it runs an entirely different belief tree construction
    if (Globals::config.AGENT == HyLEAP || Globals::config.AGENT == HyPLAN) {
		// call this also during decoupling, because we must send the root observation in order to train the NN
		network_evaluate(root);
		if (Globals::config.DECOUPLE) {
			// initialize bounds "normally", i.e. IS-DESPOT vanilla style
			InitBounds(root, lower_bound, upper_bound, streams, history);			
		} else {
			// set DRL agent's belief state value estimate as upper bound (only done when not decoupled)
			InitBoundsHyLEAP(root, lower_bound, streams, history);
		}
	}
	else {
		// initialize bounds "normally", i.e. IS-DESPOT vanilla style
		InitBounds(root, lower_bound, upper_bound, streams, history);
	}
	
	// root node initial bounds
	root_lower_bounds.push_back(initial_lower_bounds.front());
	root_upper_bounds.push_back(initial_upper_bounds.front());
	
	model->PrintBelief(root->id(), root->depth(), root->lower_bound(), root->upper_bound(), root->particles());

	// IS-DESPOT main loop, i.e. belief tree construction
	double used_time = 0;
	int num_trials = 0;
	do {
		double start = clock();
		VNode* cur = Trial(root, streams, lower_bound, upper_bound, model, history);
		Backup(cur);
		// do not add bounds of the root twice
		if (cur->depth() != 0) {
			// root node subsequent bounds
			root_lower_bounds.push_back(root->lower_bound());
			root_upper_bounds.push_back(root->upper_bound());
		}
		used_time += double(clock() - start) / CLOCKS_PER_SEC;
		num_trials++;

	} while (used_time * (num_trials + 1.0) / num_trials < timeout && gap(root) > Globals::config.TARGET_GAP);

	construct_tree_time = std::chrono::duration_cast<dsec>(Clock::now() - enter).count() * 1000;

	// trial depth
	std::pair<double, double> avg_and_std = MathUtils::get_average_and_stdev(trial_depths);
	std::pair<double, double> min_and_max = MathUtils::get_min_and_max(trial_depths);
	logd << __FUNCTION__ << endl
		 << "\t- TIME: " << construct_tree_time << endl
		 << "\t- total tree size: " << tree_nodes << endl
		 << "\t- TRIALS (num: " << trial_depths.size()
		 << ", avg depth: " << avg_and_std.first
		 << ", std depth: " << avg_and_std.second
		 << ", min depth: " << min_and_max.first
		 << ", max depth: " << min_and_max.second << ")" 
		 << endl;

	return root;
}


VNode* DESPOT::HackyConstructTree(vector<State*>& particles, 
								  RandomStreams& streams,
								  ScenarioLowerBound* lower_bound, 
								  ScenarioUpperBound* upper_bound,
								  const DSPOMDP* model, 
								  History& history, 
								  double timeout) 
{
	logd << __FUNCTION__ << endl;
	const auto enter = Clock::now();

	// assign scneraio id to particles
	for (int i = 0; i < particles.size(); i++) {
		particles[i]->scenario_id = i;
	}

	// root belief holds all particles
	VNode* root = new VNode(particles, 0);
	tree_nodes++;

	// send observation to DRLA for evaluation
	network_evaluate(root);

	// initialize bounds normally, i.e. ignore HyLEAP upper bound
	// this ensures that regardless of HyLEAP's upper bound at least one expansion happens
	// because otherwise, it is highly likely that belief tree construction terminates with only the root node
	// since it is more often than not the case that u(b0) < l(b0) -> u(b0) = l(b0) and thus we terminate immediately
    InitBounds(root, lower_bound, upper_bound, streams, history);

	// root node initial bounds
	root_lower_bounds.push_back(initial_lower_bounds.front());
	root_upper_bounds.push_back(initial_upper_bounds.front());
	logd << "\t- root bounds before consistency check:" << endl
		 << "\t\t- l(b0): " << root_lower_bounds.front() << ", u(b0): " << root_upper_bounds.front() << endl
		 << "\t- root bounds after consistency check:" << endl
		 << "\t\t- l(b0): " << root->lower_bound() << ", u(b0): " << root->upper_bound() << endl;
		 
	// call ExpandHyLEAP because we need to have the updated LSTM state for nodes at depth == 1
	int num_expanded_vnodes = ExpandHyLEAP(root, lower_bound, upper_bound, model, streams, history);

	// remove previously added vnode bounds to prevent duplicate/"wrong" entries
	initial_lower_bounds.resize(initial_lower_bounds.size() - num_expanded_vnodes);
	initial_upper_bounds.resize(initial_upper_bounds.size() - num_expanded_vnodes);

	// re-caclulate qnode bounds as well
	double qnode_lower_bound = 0;
	double qnode_upper_bound = 0;
	// for debugging
	int node_index = 1;
	logd << __FUNCTION__ << endl
		 << "\t- overwriting HyLEAP bounds at depth == 1:" << endl;
	for (QNode* qnode: root->children()) {
		qnode_lower_bound = qnode->step_reward;
		qnode_upper_bound = qnode->step_reward;
		for (auto& map_element: qnode->children()) {
			OBS_TYPE observation = map_element.first;
			VNode* vnode = map_element.second;

			// override HyLEAP's upper bound (NN belief state estimate) with the regular upper bound heuristic 
			// of IS-DESPOT (minimum number of steps until goal position has been reached) for depth == 1
			history.Add(qnode->edge(), observation);
			InitBounds(vnode, lower_bound, upper_bound, streams, history);

			// print debugging information
			logd << "\t\t-- l(b" << node_index << "): " << vnode->lower_bound() 
				 << ", u(b" << node_index << "): " << vnode->upper_bound() 
				 << ", e(b" << node_index << "): " << gap(vnode) 
				 << ", E(b" << node_index << "): " << weu(vnode)
				 << endl;

			history.RemoveLast();
			qnode_lower_bound += vnode->lower_bound();
			qnode_upper_bound += vnode->upper_bound();
			node_index++;
		}
		// finalize qnode bounds
		qnode->lower_bound(qnode_lower_bound);
		qnode->upper_bound(qnode_upper_bound);
		qnode->utility_upper_bound = qnode_upper_bound + Globals::config.PRUNING_CONSTANT;
		qnode->default_value = qnode_lower_bound; // for debugging
	}

	// iterative tree construction
	int num_root_returns = 0;
	do {
		// always expands accelerate for depth = 0 
		VNode* cur = HackyTrial(root, streams, lower_bound, upper_bound, model, history);

		// this happens when the gap at the root is 0
		// in other words: as long as the gap of the root node is 0, we won't terminate (until 100ms are exhausted)
        if (cur == root) {
			num_root_returns = 0;
		// this is the general case, i.e. trials are still changing the gap of the root node
		// thus: if 5 consecutive trials have changed the gap at the ropot node, we terminate
		} else {
			++num_root_returns;
		}
        
		Backup(cur);

		// do not add bounds of the root twice
		if (cur->depth() != 0) {
			// root node subsequent bounds
			root_lower_bounds.push_back(root->lower_bound());
			root_upper_bounds.push_back(root->upper_bound());
		}
	// quit after 100 ms or when trials stopped at depth = 0 for 5 times in a row
	} while (std::chrono::duration_cast<dsec>(Clock::now() - enter).count()*1000 <= 100 && num_root_returns < 5);

	construct_tree_time = std::chrono::duration_cast<dsec>(Clock::now() - enter).count() * 1000;

	// trial depth
	std::pair<double, double> avg_and_std = MathUtils::get_average_and_stdev(trial_depths);
	std::pair<double, double> min_and_max = MathUtils::get_min_and_max(trial_depths);
	logd << __FUNCTION__ << endl
		 << "\t- l(b0): " << root_lower_bounds.back() << ", u(b0): " << root_upper_bounds.back() << endl
		 << "\t- num_root_returns: " << num_root_returns << endl
		 << "\t- total number of trials: " << trial_depths.size() << endl
		 << "\t- average trial depth: " << avg_and_std.first << endl
		 << "\t- trial depth standard deviation: " << avg_and_std.second << endl
		 << "\t- min trial depth: " << min_and_max.first << endl
		 << "\t- max trial depth: " << min_and_max.second << endl
		 << "\t- total tree size: " << tree_nodes << endl; 

	return root;
}


VNode* DESPOT::Trial(VNode* root, 
					 RandomStreams& streams,
					 ScenarioLowerBound* lower_bound, 
					 ScenarioUpperBound* upper_bound,
					 const DSPOMDP* model, 
					 History& history) 
{
	const auto enter = Clock::now();
	VNode* cur = root;
	int hist_size = history.Size();

	int terminal_states_at_trial_start = 0;
	do {
		terminal_states_at_trial_start = terminal_states;
		exploit_blocker(cur);

		if (gap(cur) == 0) { break; }

		// only expand when current node is a leaf
		if (cur->IsLeaf()) {
			// only HyLEAP and HyPLAN require a special bound initialization
			// "hacky" will never reach here, because it runs an entirely different belief tree construction
			// so we only have to worry about "decouple"
    		if ((Globals::config.AGENT == HyLEAP || Globals::config.AGENT == HyPLAN) && !Globals::config.DECOUPLE) {
				ExpandHyLEAP(cur, lower_bound, upper_bound, model, streams, history);
			} else {
				Expand(cur, lower_bound, upper_bound, model, streams, history);
			}
		}

		QNode* qstar = select_best_upper_bound_node(cur);
		VNode* next = select_best_weu_node(qstar);
		if (next == NULL) { break; }

		cur = next;
		history.Add(qstar->edge(), cur->edge());

	} while (cur->depth() < Globals::config.MAX_SEARCH_DEPTH && weu(cur) > 0);

	const auto single_trial_time = (std::chrono::duration_cast<dsec>(Clock::now() - enter).count() * 1000);
	trial_time += single_trial_time;
	// track trial lengths
	trial_depths.push_back(cur->depth()* 1.0);

	logd << __FUNCTION__ << endl
		 << "\t- #TRIAL: " << trial_depths.size() << endl
		 << "\t\t- TIME: " << single_trial_time << "ms" << endl
		 << "\t\t- DEPTH: " << trial_depths.back() << endl
		 << "\t\t- TRACE: " << history << endl
		 << "\t\t- TREE SIZE: " << tree_nodes << endl
		 << "\t\t- #TERMINAL: " << terminal_states - terminal_states_at_trial_start << endl;

	// if current is not the root node (can happen when all child belief nodes are terminal states)
	// e.g. last step just before reaching the goal position
	if (cur->depth() != 0) {
		// get all observations following the last expanded action
		for (const auto& sibling: GetValues(cur->parent()->children())) {
			model->PrintBelief(
				sibling->id(), sibling->depth(), sibling->lower_bound(), sibling->upper_bound(), sibling->particles()
			);
		}
	}
	history.Truncate(hist_size);
	return cur;
}


VNode* DESPOT::HackyTrial(VNode* root, 
						  RandomStreams& streams,
						  ScenarioLowerBound* lower_bound, 
						  ScenarioUpperBound* upper_bound,
						  const DSPOMDP* model, 
						  History& history) 
{
	const auto enter = Clock::now();
	VNode* cur = root;
	int hist_size = history.Size();

    do {
    	exploit_blocker(cur);

    	if (gap(cur) == 0) { break; }

		// only expand nodes that do not have children yet
		// this prevents duplciate expansions since we already expanded once "normally" beforehand
    	if (cur->IsLeaf()) {
			ExpandHyLEAP(cur, lower_bound, upper_bound, model, streams, history);
		} 
    
    	QNode* qstar = nullptr;
		// always take accelerate action at depth = 0 ...
    	if (cur->depth() == 0) {
			qstar = root->children()[ModelParams::ACT_ACC];
		} else {
			qstar = select_best_upper_bound_node(cur);
		}

    	VNode* next = select_best_weu_node(qstar);
    	if (next == NULL) {
			break;
		}
    	cur = next;

    	history.Add(qstar->edge(), cur->edge());
    } while (cur->depth() < Globals::config.MAX_SEARCH_DEPTH && weu(cur) > 0);

	// track trial lengths
	trial_depths.push_back(cur->depth()* 1.0);
    history.Truncate(hist_size);

	trial_time += (std::chrono::duration_cast<dsec>(Clock::now() - enter).count() * 1000);

    return cur;
}


void DESPOT::Expand(VNode* vnode,
					ScenarioLowerBound* lower_bound, 
					ScenarioUpperBound* upper_bound,
					const DSPOMDP* model, 
					RandomStreams& streams,
					History& history) 
{
	const auto enter = Clock::now();
	int num_expanded_vnodes = 0;
	vector<QNode*>& children = vnode->children();
	for (ACT_TYPE action = 0; action < model->NumActions(); action++) {

		auto enter = Clock::now();
		QNode* qnode = new QNode(vnode, action);
		qcreation_time += (std::chrono::duration_cast<dsec>(Clock::now() - enter).count() * 1000);

		children.push_back(qnode);
		Expand(qnode, lower_bound, upper_bound, model, streams, history);
		num_expanded_vnodes += qnode->children().size();
	}
	expand_time += (std::chrono::duration_cast<dsec>(Clock::now() - enter).count() * 1000);
}


void DESPOT::Expand(QNode* qnode, 
					ScenarioLowerBound* lb,
					ScenarioUpperBound* ub, 
					const DSPOMDP* model,
					RandomStreams& streams,
					History& history) 
{
	VNode* parent = qnode->parent();
	streams.position(parent->depth());
	map<OBS_TYPE, VNode*>& children = qnode->children();

	const vector<State*>& particles = parent->particles();

	double step_reward = 0;

	// Partition particles by observation
	map<OBS_TYPE, vector<State*>> partitions;
	OBS_TYPE obs;
	double reward;
	// create an observation for eahc particle
	for (int i = 0; i < particles.size(); i++) {
		State* particle = particles[i];

		auto enter = Clock::now();
		State* copy = model->Copy(particle);
		copy_time += (std::chrono::duration_cast<dsec>(Clock::now() - enter).count() * 1000);

		bool terminal;

		enter = Clock::now();
		// LEADER doesn't change importance weight during simulation
		if (!Globals::config.NO_IMPORTANCE_SAMPLING && Globals::config.AGENT != LEADER) {
			terminal = model->ImportanceSamplingStep(*copy, streams.Entry(copy->scenario_id), qnode->edge(), reward, obs);
		} else {
			terminal = model->Step(*copy, streams.Entry(copy->scenario_id), qnode->edge(), reward, obs);
		}
		step_time += (std::chrono::duration_cast<dsec>(Clock::now() - enter).count() * 1000);

		step_reward += reward * particle->weight;
		if (!terminal) {
			// group states by observations
			// i.e. scenarios that resulted in the same pomdp state (without the ped's intention)
			partitions[obs].push_back(copy);
		} else {
			enter = Clock::now();
			model->Free(copy);
			free_time += (std::chrono::duration_cast<dsec>(Clock::now() - enter).count() * 1000);
			terminal_states++;
		}
	}
	// how many observations have been created?
	observations.push_back(partitions.size() * 1.0);

	auto enter = Clock::now();
	// normalize weight: only relevant if we actually use importance sampling, have not disabled normalization
	// and do not use LEADER, because it is normalized once for the root node and then never changed again
	if(!Globals::config.NO_IMPORTANCE_SAMPLING && !Globals::config.NO_NORMALIZATION && Globals::config.AGENT != LEADER) {
		// compute weight of the parent
		double parent_weight = parent -> Weight();
		// compute weight of the particles for qnode, i.e. the weight of the children
		double children_weight = 0;
		for (map<OBS_TYPE, vector<State*>>::iterator it = partitions.begin(); it != partitions.end(); it++) {
			OBS_TYPE obs = it->first;
			children_weight += partitions[obs][0]->Weight(partitions[obs]);
		}
		double normization_constant;
		normization_constant = (children_weight==0) ? 1 : parent_weight/children_weight;

		if (normization_constant != 1) {
			for (map<OBS_TYPE, vector<State*>>::iterator it = partitions.begin(); it != partitions.end(); it++) {
				OBS_TYPE obs = it->first;
				for(int i = 0; i < partitions[obs].size(); i++){
					partitions[obs][i]->weight = partitions[obs][i]->weight * normization_constant;
				}
			}			
		}
	}
	norm_time += (std::chrono::duration_cast<dsec>(Clock::now() - enter).count() * 1000);

	step_reward = Globals::Discount(parent->depth()) * step_reward - Globals::config.PRUNING_CONSTANT;
	double lower_bound = step_reward;
	double upper_bound = step_reward;

	// create new belief nodes
	for (map<OBS_TYPE, vector<State*>>::iterator it = partitions.begin(); it != partitions.end(); it++) {
		OBS_TYPE obs = it->first;
		auto enter = Clock::now();
		VNode* vnode = new VNode(partitions[obs], tree_nodes, parent->depth() + 1, qnode, obs);
		vcreation_time += (std::chrono::duration_cast<dsec>(Clock::now() - enter).count() * 1000);

		tree_nodes++;

		children[obs] = vnode;
		history.Add(qnode->edge(), obs);
		InitBounds(vnode, lb, ub, streams, history);
		history.RemoveLast();

		lower_bound += vnode->lower_bound();
		upper_bound += vnode->upper_bound();
	}

	qnode->step_reward = step_reward;
	qnode->lower_bound(lower_bound);
	qnode->upper_bound(upper_bound);
	qnode->utility_upper_bound = upper_bound + Globals::config.PRUNING_CONSTANT;
	qnode->default_value = lower_bound; // for debugging
}


int DESPOT::ExpandHyLEAP(VNode* parent,
						 ScenarioLowerBound* lower_bound, 
						 ScenarioUpperBound* upper_bound,
						 const DSPOMDP* model, 
						 RandomStreams& streams,
						 History& history) 
{
	if (despot::Globals::config.DECOUPLE) {
		printf("IS-DESPOT::[%s] Communication with NN not allowed during decoupling.\n", __PRETTY_FUNCTION__);
		exit(-1);
	}

	logd << __FUNCTION__ << endl;
	const auto enter = Clock::now();

	// how many vnodes have been created for each qnode?
	std::vector<int> vnodes_per_qnode;
	// variable used for delayed bound initialization
	std::vector<VNode*> expanded_nodes;

	int num_vnodes = 0;
	vector<QNode*>& children = parent->children();
	for (ACT_TYPE action = 0; action < model->NumActions(); action++) {
		// track time
		auto enter = Clock::now();
		QNode* qnode = new QNode(parent, action);
		qcreation_time += (std::chrono::duration_cast<dsec>(Clock::now() - enter).count() * 1000);

		children.push_back(qnode);
		// expand with delayed bound initialization n order to save tcp-communication overhead
		num_vnodes = ExpandHyLEAP(qnode, lower_bound, upper_bound, model, streams, history, expanded_nodes);
		// used for indexing during bound initialization
		vnodes_per_qnode.push_back(num_vnodes);
	}

	// evaluate all newly expanded belief nodes in one swoop
	if (!expanded_nodes.empty()) {
		logd << "\t- expanded vnodes: " << expanded_nodes.size() 
		     << " at depth == " << expanded_nodes.front()->depth() << ":" << endl;
		network_evaluate(expanded_nodes);

		int vnode_idx_offset = 0;
		int print_idx = 0;
		// for each qnode
		for (ACT_TYPE action = 0; action < model->NumActions(); action++) {
			QNode* qnode = children[action];

			// init bounds of qnode
			double qnode_lower_bound = qnode->step_reward;
			double qnode_upper_bound = qnode->step_reward;
			
			if (action == 1) {vnode_idx_offset += vnodes_per_qnode[0];}
			if (action == 2) {vnode_idx_offset += vnodes_per_qnode[1];}
			// for each vnode of each qnode
			for (int vnode_idx = 0; vnode_idx < vnodes_per_qnode[static_cast<int>(action)]; vnode_idx++) {
				// get vnode
				VNode* vnode = expanded_nodes[vnode_idx];
				// update history
				history.Add(qnode->edge(), vnode->edge());
				// init bounds
				InitBoundsHyLEAP(vnode, lower_bound, streams, history);

				// print debugging information
				logd << "\t- l(b" << tree_nodes - expanded_nodes.size() + print_idx << "): " << vnode->lower_bound() 
				 	 << ", u(b" << tree_nodes - expanded_nodes.size() + print_idx  << "): " << vnode->upper_bound() 
				 	 << ", e(b" << tree_nodes - expanded_nodes.size() + print_idx  << "): " << gap(vnode) 
				 	 << ", E(b" << tree_nodes - expanded_nodes.size() + print_idx  << "): " << weu(vnode)
					 << endl;
				print_idx++;

				history.RemoveLast();
				// update qnode bounds
				qnode_lower_bound += vnode->lower_bound();
				qnode_upper_bound += vnode->upper_bound();
			}

			// finalize qnode bounds
			qnode->lower_bound(qnode_lower_bound);
			qnode->upper_bound(qnode_upper_bound);
			qnode->utility_upper_bound = qnode_upper_bound + Globals::config.PRUNING_CONSTANT;
			qnode->default_value = qnode_lower_bound; // for debugging
		}
	}
	expand_time += (std::chrono::duration_cast<dsec>(Clock::now() - enter).count() * 1000);
	return expanded_nodes.size();

}


// expand with delayed bound initialization, i.e. all newly expanded belief nodes are send
// to the hyleap network in one swoop
int DESPOT::ExpandHyLEAP(QNode* qnode, 
						 ScenarioLowerBound* lb,
						 ScenarioUpperBound* ub, 
						 const DSPOMDP* model,
						 RandomStreams& streams,
						 History& history,
						 std::vector<VNode*>& expanded_nodes) 
{
	if (despot::Globals::config.DECOUPLE) {
		printf("IS-DESPOT::[%s] Communication with NN not allowed during decoupling.\n", __PRETTY_FUNCTION__);
		exit(-1);
	}

	VNode* parent = qnode->parent();
	streams.position(parent->depth());
	map<OBS_TYPE, VNode*>& children = qnode->children();
	const vector<State*>& particles = parent->particles();

	// partition particles by observation
	map<OBS_TYPE, vector<State*>> partitions;
	OBS_TYPE obs;
	// reset rewards
	double step_reward = 0, reward = 0;
	// create an observation for eahc particle
	for (int i = 0; i < particles.size(); i++) {
		State* particle = particles[i];

		auto enter = Clock::now();
		State* copy = model->Copy(particle);
		copy_time += (std::chrono::duration_cast<dsec>(Clock::now() - enter).count() * 1000);

		bool terminal = false;
		enter = Clock::now();
		if(!Globals::config.NO_IMPORTANCE_SAMPLING) {
			terminal = model->ImportanceSamplingStep(*copy, streams.Entry(copy->scenario_id), qnode->edge(), reward, obs);
		} else {
			terminal = model->Step(*copy, streams.Entry(copy->scenario_id), qnode->edge(), reward, obs);
		}
		step_time += (std::chrono::duration_cast<dsec>(Clock::now() - enter).count() * 1000);

		step_reward += reward * particle->weight;

		if (!terminal) {
			// group states by observations
			// i.e. scenarios that resulted in the same pomdp state (without the ped's intention)
			partitions[obs].push_back(copy);
		} else {
			enter = Clock::now();
			model->Free(copy);
			free_time += (std::chrono::duration_cast<dsec>(Clock::now() - enter).count() * 1000);
			terminal_states++;
		}
	}

	// how many observations have been created?
	observations.push_back(partitions.size() * 1.0);

	auto enter = Clock::now();
	// normalize the weight
	if(!Globals::config.NO_IMPORTANCE_SAMPLING && !Globals::config.NO_NORMALIZATION) {
		//compute the weight of the parent
		double parent_weight = parent -> Weight();
		//compute the weight of the particles for qnode, i.e., the weight of the children
		double children_weight = 0;
		for (map<OBS_TYPE, vector<State*> >::iterator it = partitions.begin();
			it != partitions.end(); it++) {
			OBS_TYPE obs = it->first;
			children_weight += partitions[obs][0]->Weight(partitions[obs]);
		}
		
		double normization_constant = (children_weight==0) ? 1 : parent_weight/children_weight;
		if (normization_constant != 1){
			for (map<OBS_TYPE, vector<State*>>::iterator it = partitions.begin(); it != partitions.end(); it++) {
				OBS_TYPE obs = it->first;
				for(int i=0; i<partitions[obs].size(); i++){
					partitions[obs][i]->weight = partitions[obs][i]->weight * normization_constant;
				}
			}
		}
	}
	norm_time += (std::chrono::duration_cast<dsec>(Clock::now() - enter).count() * 1000);

	// will be the base for calculating the qnode bounds later on
	qnode->step_reward = Globals::Discount(parent->depth()) * step_reward
					   // pruning_constant is used for regularization
					   - Globals::config.PRUNING_CONSTANT;

	// create new belief nodes but do not initiate their bounds yet
	for (map<OBS_TYPE, vector<State*>>::iterator it = partitions.begin(); it != partitions.end(); it++) {
		OBS_TYPE obs = it->first;

		auto enter = Clock::now();
		VNode* vnode = new VNode(partitions[obs], tree_nodes, parent->depth() + 1, qnode, obs);
		vcreation_time += (std::chrono::duration_cast<dsec>(Clock::now() - enter).count() * 1000);

		tree_nodes++;

		children[obs] = vnode;
		expanded_nodes.push_back(vnode);
		// SKIP INITIALIZATION OF NEWLY CREATED BELIEF NODES
	}
	// number of vnodes created
	return partitions.size();
}


void DESPOT::Backup(VNode* vnode) 
{
	const auto enter = Clock::now();
	while (true) {
		Update(vnode);

		QNode* parentq = vnode->parent();

		if (parentq == NULL) 
			break;
		
		Update(parentq);

		vnode = parentq->parent();

		backups++;
	}
	const auto single_backup_time = (std::chrono::duration_cast<dsec>(Clock::now() - enter).count() * 1000);
	backup_time += single_backup_time;

	// display bound changes at the level of the root node
	// vnode will always be the root node at last
	logd << __FUNCTION__ << endl
		 << "\t\t- TIME: [" << single_backup_time << "ms]" << endl
		 << "\t\t- l(b0): " << vnode->lower_bound() << ", u(b0): " << vnode->upper_bound() << endl;
	// print bounds of root-children
	for (ACT_TYPE action = 0; action < vnode->children().size(); action++) {
		QNode* qnode = vnode->Child(action);
		logd << "\t\t- l(b0, " << ModelParams::action_idx_to_string[action] << "): " << qnode->lower_bound() << endl;
	}
}


void DESPOT::Update(VNode* vnode) 
{
	if (vnode->IsLeaf()) 
		return;
	
	double lower = vnode->default_move().value;
	double upper = vnode->default_move().value;
	double utility_upper = Globals::NEG_INFTY;

	for (ACT_TYPE action = 0; action < vnode->children().size(); action++) {
		QNode* qnode = vnode->Child(action);

		// "qnode->lower_bound()" reflects the second argument of the outer maximization,
		// i.e. the step reward associated with that action + the sum of the updated lower
		// bounds of all children/ observation
		lower = max(lower, qnode->lower_bound());
		upper = max(upper, qnode->upper_bound());
		utility_upper = max(utility_upper, qnode->utility_upper_bound);
	}

	// lower bound improvements are propagated
	if (lower > vnode->lower_bound()) 
		vnode->lower_bound(lower);

	// upper bound regressions are propagated
	if (upper < vnode->upper_bound()) 
		vnode->upper_bound(upper);

	if (utility_upper < vnode->utility_upper_bound) {
		vnode->utility_upper_bound = utility_upper;
	}
}


void DESPOT::Update(QNode* qnode) 
{
	double lower = qnode->step_reward;
	double upper = qnode->step_reward;
	double utility_upper = qnode->step_reward + Globals::config.PRUNING_CONSTANT;

	map<OBS_TYPE, VNode*>& children = qnode->children();
	for (map<OBS_TYPE, VNode*>::iterator it = children.begin(); it != children.end(); it++) {
		VNode* vnode = it->second;

		// backup-ed bound is RWDU + updated bound value of all children/following observations
		// bound balue of children might be initial one
		lower += vnode->lower_bound();
		upper += vnode->upper_bound();
		utility_upper += vnode->utility_upper_bound;
	}
	// lower bound improvements are propagated
	// "lower" contains updated lower bound values of all children, while
	// "lower_bound" contains only the initial lower bound values of all children
	if (lower > qnode->lower_bound()) { 
		qnode->lower_bound(lower);
	}
	// upper bound regressions are propagated
	if (upper < qnode->upper_bound()) { 
		qnode->upper_bound(upper);
	}	
	if (utility_upper < qnode->utility_upper_bound) {
		qnode->utility_upper_bound = utility_upper;
	}
}


void DESPOT::InitLowerBound(VNode* vnode, 
							ScenarioLowerBound* lower_bound,
							RandomStreams& streams, 
							History& history) 
{
	streams.position(vnode->depth());
	ValuedAction Lb = lower_bound->Value(vnode->particles(), streams, history);
	if (Globals::config.AGENT == HyPLAN && !Globals::config.NO_VERTICAL_PRUNING) {
		ValuedAction Ub = vnode->drla_prediction;
		// uncertainty weighted lower bound
		Lb.value = (Ub.uncertainty*Lb.value + (1 - Ub.uncertainty)*Ub.value);
		uncertainty_values.push_back(Ub.uncertainty);
	}
	Lb.value *= Globals::Discount(vnode->depth());
	vnode->default_move(Lb);
	vnode->lower_bound(Lb.value);
}


void DESPOT::InitUpperBound(VNode* vnode, 
							ScenarioUpperBound* upper_bound,
							RandomStreams& streams, 
							History& history) 
{
	streams.position(vnode->depth());
	double upper = upper_bound->Value(vnode->particles(), streams, history);
	vnode->utility_upper_bound = upper * Globals::Discount(vnode->depth());
	vnode->upper_bound(vnode->utility_upper_bound - Globals::config.PRUNING_CONSTANT);
}


void DESPOT::InitBounds(VNode* vnode, 
						ScenarioLowerBound* lower_bound,
						ScenarioUpperBound* upper_bound, 
						RandomStreams& streams, 
						History& history) 
{

	const auto enter = Clock::now();
	InitLowerBound(vnode, lower_bound, streams, history);
	InitUpperBound(vnode, upper_bound, streams, history);

	// log initial bound values
	initial_lower_bounds.push_back(vnode->lower_bound());
	initial_upper_bounds.push_back(vnode->upper_bound());

	// close gap because no more search can be done on leaf node
	if (vnode->upper_bound() < vnode->lower_bound() || vnode->depth() == Globals::config.MAX_SEARCH_DEPTH - 1) {
		vnode->upper_bound(vnode->lower_bound());
	}
	init_bounds_time += (std::chrono::duration_cast<dsec>(Clock::now() - enter).count() * 1000);
}


void DESPOT::InitUpperBoundHyLEAP(VNode* vnode)
{
	// not weighted by number of particles
	// completely disregards importance distribution
	vnode->utility_upper_bound = vnode->drla_prediction.value * Globals::Discount(vnode->depth());
	vnode->upper_bound(vnode->utility_upper_bound - Globals::config.PRUNING_CONSTANT);
}


void DESPOT::InitBoundsHyLEAP(VNode* vnode, 
							  ScenarioLowerBound* lower_bound, 
							  RandomStreams& streams, 
							  History& history) 
{
	if (despot::Globals::config.DECOUPLE) {
		printf("IS-DESPOT::[%s] Using NN value estimate as uppoer bouund is not allowed during decoupling.\n",
			   __PRETTY_FUNCTION__);
		exit(-1);
	}
	const auto enter = Clock::now();

	InitLowerBound(vnode, lower_bound, streams, history);
	InitUpperBoundHyLEAP(vnode);

	// log initial bound values
	initial_lower_bounds.push_back(vnode->lower_bound());
	initial_upper_bounds.push_back(vnode->upper_bound());

	// close gap because no more search can be done on leaf node
	if (vnode->upper_bound() < vnode->lower_bound() || vnode->depth() == Globals::config.MAX_SEARCH_DEPTH - 1) {
		vnode->upper_bound(vnode->lower_bound());
	}
	init_bounds_time += (std::chrono::duration_cast<dsec>(Clock::now() - enter).count() * 1000);
}


double DESPOT::gap(VNode* vnode) {
	return (vnode->upper_bound() - vnode->lower_bound());
}


double DESPOT::weu(VNode* vnode) {
	return weu(vnode, Globals::config.GAP_REDUCTION_RATE);
}


// can pass root as an argument, but will not affect performance much
double DESPOT::weu(VNode* vnode, double xi) {
	VNode* root = vnode;
	while (root->parent() != NULL) {
		root = root->parent()->parent();
	}
	double weu = gap(vnode) - xi * vnode->Weight() * gap(root);
	excess_uncertainty.push_back(weu);
	return weu;
}


VNode* DESPOT::select_best_weu_node(QNode* qnode) {
	double weustar = Globals::NEG_INFTY;
	VNode* vstar = NULL;
	map<OBS_TYPE, VNode*>& children = qnode->children();
	for (map<OBS_TYPE, VNode*>::iterator it = children.begin();
		it != children.end(); it++) {
		VNode* vnode = it->second;

		double weu = DESPOT::weu(vnode);
		if (weu >= weustar) {
			weustar = weu;
			vstar = vnode->vstar;
		}
	}
	return vstar;
}


QNode* DESPOT::select_best_upper_bound_node(VNode* vnode) {
	int astar = -1;
	double upperstar = Globals::NEG_INFTY;
	for (ACT_TYPE action = 0; action < vnode->children().size(); action++) {
		QNode* qnode = vnode->Child(action);

		// greater OR EQUALS favors actions that come later in case of a tie, i.e. accelerate
		if (qnode->upper_bound() >= upperstar) {
			upperstar = qnode->upper_bound();
			astar = action;
		}
	}
	if (astar < 0) {
		printf("IS-DESPOT::[%s] Invalid action selected: Expected one of {0, 1, 2}, got %d.\n", __PRETTY_FUNCTION__, astar);
		exit(-1);
	}
	return vnode->Child(astar);
}


ScenarioLowerBound* DESPOT::lower_bound() const {
	return lower_bound_;
}


ScenarioUpperBound* DESPOT::upper_bound() const {
	return upper_bound_;
}


void DESPOT::exploit_blocker(VNode* vnode) {
	// is pruning enabled?
	if (Globals::config.PRUNING_CONSTANT <= 0) {
		return;
	}
	VNode* cur = vnode;
	while (cur != NULL) {
		VNode* blocker = find_blocker(cur);

		// if a blocker exists
		if (blocker != NULL) {
			// if root or current node is blocker
			if (cur->parent() == NULL || blocker == cur) {
				// MakeDefault procedurel, i.e. set gap == 0 and terminate planning trial
				// only blocks cur for future planning trials
				double value = cur->default_move().value;
				cur->lower_bound(value);
				cur->upper_bound(value);
				cur->utility_upper_bound = value;
			} else {
				// for all observations following the same action as the current one at the same depth of the tree
				const map<OBS_TYPE, VNode*>& siblings = cur->parent()->children();
				// close gap for all observations, i.e. block entire subtree for future planning trials
				for (map<OBS_TYPE, VNode*>::const_iterator it = siblings.begin(); it != siblings.end(); it++) {
					VNode* node = it->second;
					double value = node->default_move().value;
					node->lower_bound(value);
					node->upper_bound(value);
					node->utility_upper_bound = value;
				}
			}

			Backup(cur);

			if (cur->parent() == NULL) {
				cur = NULL;
			} else {
				cur = cur->parent()->parent();
			}
		} else {
			break;
		}
	}
}


VNode* DESPOT::find_blocker(VNode* vnode) {
	VNode* cur = vnode;
	int count = 1;
	while (cur != NULL) {
		// is current vnode blocker?
		// utility_upper_bound is weighted and discounted value returned by upper bound heuiristic, i.e. non-regularized
		// any ancestor node for which the regularazitation-adjusted upper bound is equal or lower than its lower bound is a blocker
		// the upper bound of the ancestor belief b' is reduced linearly with growing distance to the current belief b by a constant factor (i.e. the pruning constant)
		// in other words: the regularization-adjusted gap is 0 for a blocker node
		if (cur->utility_upper_bound - count * Globals::config.PRUNING_CONSTANT <= cur->default_move().value) {
			break;
		}
		count++;
		// there is no blocker
		if (cur->parent() == NULL) {
			cur = NULL;
		// get vnode parent
		} else {
			cur = cur->parent()->parent();
		}
	}
	return cur;
}


ValuedAction DESPOT::compute_optimal_action(VNode* vnode) {
	logd << __FUNCTION__ << endl;

	ValuedAction astar(-1, Globals::NEG_INFTY);
	for (ACT_TYPE action = 0; action < vnode->children().size(); action++) {
		QNode* qnode = vnode->Child(action);
		// ""="" favors MAIN over DEC or ACC over MAIN in a tie
		if (qnode->lower_bound() >= astar.value) {
			astar = ValuedAction(action, qnode->lower_bound());
		}
	}

	double root_default_move_value = vnode->default_move().value;
	logd << "\t- root default action: " << ModelParams::action_idx_to_string[vnode->default_move().action] 
		 << " with l(b): " << root_default_move_value << endl;
	
	if (root_default_move_value > astar.value) {
		astar = vnode->default_move();
	}

	// hacky stuff...
	if ((Globals::config.AGENT == HyLEAP || Globals::config.AGENT == HyPLAN) && Globals::config.HACKY) {
		if(abs(vnode->upper_bound() - vnode->lower_bound()) < 0.1 && vnode->upper_bound() <= -0.5)
			// decelerate
			astar.action = ModelParams::ACT_DEC;
	}
	logd << "\t- optimal action: " << ModelParams::action_idx_to_string[astar.action] 
		 << " with l(b0, a): " << astar.value << endl;

	return astar;
}

} // namespace despot

