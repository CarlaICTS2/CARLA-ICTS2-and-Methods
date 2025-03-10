/*
 * particle_belief.cpp
 *
 *  Created on: 21 Sep 2017
 *      Author: panpan
 */

#include <despot/core/particle_belief.h>
#include <despot/interface/pomdp.h>
#include "../../carla_car/src/state.h"



using namespace std;

namespace despot {

/* =============================================================================
 * ParticleBelief class
 * =============================================================================*/

ParticleBelief::ParticleBelief(vector<State*> particles, const DSPOMDP* model, Belief* prior, bool split) :
	Belief(model), 
	particles_(particles), 
	num_particles_(particles.size()), 
	prior_(prior), 
	split_(split), 
	state_indexer_(NULL) 
{
	// check for proper probability distribution
	if (fabs(State::Weight(particles) - 1.0) > 1e-6 && !Globals::config.NO_NORMALIZATION) {
		printf("IS-DESPOT::[%s] Invalid probability distribution: " 
				"Expected total probability mass of 1, got %.4f.\n", __PRETTY_FUNCTION__, State::Weight(particles));
		exit(-1);   
	}

	if (split) {
		// Maintain more particles to avoid degeneracy
		while (2 * num_particles_ < 5000)
			num_particles_ *= 2;
		if (particles_.size() < num_particles_) {
			vector<State*> new_particles;
			int n = num_particles_ / particles_.size();
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < particles_.size(); j++) {
					State* particle = particles_[j];
					State* copy = model_->Copy(particle);
					copy->weight /= n;
					new_particles.push_back(copy);
				}
			}

			for (int i = 0; i < particles_.size(); i++)
				model_->Free(particles_[i]);

			particles_ = new_particles;
		}
	}

	if (fabs(State::Weight(particles) - 1.0) > 1e-6 && !Globals::config.NO_NORMALIZATION) {
		printf("IS-DESPOT::[%s] Invalid probability distribution: " 
				"Expected total probability mass of 1, got %.4f.\n", __PRETTY_FUNCTION__, State::Weight(particles));
		exit(-1);   
	}

	random_shuffle(particles_.begin(), particles_.end());

	if (prior_ == NULL) {
		for (int i = 0; i < particles.size(); i++)
			initial_particles_.push_back(model_->Copy(particles[i]));
	}
}

ParticleBelief::~ParticleBelief() {
	for (int i = 0; i < particles_.size(); i++) {
		model_->Free(particles_[i]);
	}

	for (int i = 0; i < initial_particles_.size(); i++) {
		model_->Free(initial_particles_[i]);
	}
}

void ParticleBelief::state_indexer(const StateIndexer* indexer) {
	state_indexer_ = indexer;
}

const vector<State*>& ParticleBelief::particles() const {
	return particles_;
}

vector<State*> ParticleBelief::Sample(int num) const {
	return Sample(num, particles_, model_);
}

void ParticleBelief::Update(ACT_TYPE action, OBS_TYPE obs) {
	history_.Add(action, obs);

	vector<State*> updated;
	double total_weight = 0;
	double reward;
	OBS_TYPE o;
	// Update particles
	for (int i = 0; i <particles_.size(); i++) {
		State* particle = particles_[i];
		bool terminal = model_->Step(*particle, Random::RANDOM.NextDouble(),
			action, reward, o);
		double prob = model_->ObsProb(obs, *particle, action);

		if (!terminal && prob) { // Terminal state is not required to be explicitly represented and may not have any observation
			particle->weight *= prob;
			total_weight += particle->weight;
			updated.push_back(particle);
		} else {
			model_->Free(particle);
		}
	}

	particles_ = updated;

	// Resample if the particle set is empty
	if (particles_.size() == 0) {
		logb << "Particle set is empty!" << endl;
		if (prior_ != NULL) {
			logb << "Resampling by drawing random particles from prior which are consistent with history" << endl;
			particles_ = Resample(num_particles_, *prior_, history_);
		} else {
			logb << "Resampling by searching initial particles which are consistent with history" << endl;
			particles_ = Resample(num_particles_, initial_particles_, model_, history_);
		}

		if (particles_.size() == 0 && state_indexer_ != NULL) {
			logb << "Resampling by searching states consistent with last (action, observation) pair" << endl;
			particles_ = Resample(num_particles_, model_, state_indexer_, action, obs);
		}

		if (particles_.size() == 0) {
			logb << "Resampling failed - Using initial particles" << endl;
			for (int i = 0; i < initial_particles_.size(); i ++)
				particles_.push_back(model_->Copy(initial_particles_[i]));
		}

		//Update total weight so that effective number of particles are computed correctly
        total_weight = 0;
        for (int i = 0; i < particles_.size(); i++)
            total_weight += particles_[i]->weight;
	}


	double weight_square_sum = 0;
	for (int i = 0; i < particles_.size(); i++) {
		State* particle = particles_[i];
		particle->weight /= total_weight;
		weight_square_sum += particle->weight * particle->weight;
	}

	// Resample if the effective number of particles is "small"
	double num_effective_particles = 1.0 / weight_square_sum;
	if (num_effective_particles < num_particles_ / 2.0) {
		vector<State*> new_belief = Sample(num_particles_, particles_,
			model_);
		for (int i = 0; i < particles_.size(); i++)
			model_->Free(particles_[i]);

		particles_ = new_belief;
	}
}

Belief* ParticleBelief::MakeCopy() const {
	vector<State*> copy;
	for (int i = 0; i < particles_.size(); i++) {
		copy.push_back(model_->Copy(particles_[i]));
	}

	return new ParticleBelief(copy, model_, prior_, split_);
}

string ParticleBelief::print() const 
{
	std::ostringstream oss;

	std::map<int, double> goal_angles_to_weight;
	std::map<int, int> goal_angles_to_particle_count;
	for (int i = 0; i < particles_.size(); i++) {
		int goal_angle_degree = static_cast<PomdpState*>(particles_[i])->peds[0].goal*2;
		goal_angles_to_weight[goal_angle_degree] += particles_[i]->weight;
		goal_angles_to_particle_count[goal_angle_degree]++;
	}

	oss << "\t\t- PDF for " << particles_.size() << " particles:" << endl;
	vector<pair<int, double>> pairs = SortByValue(goal_angles_to_weight);
	for (int i = 0; i < pairs.size(); i++) {
		pair<int, double> pair = pairs[i];
		oss << "\t\t\t- pedestrian goal angle [DEG]: " << pair.first 
			<< ", probability: " << pair.second 
			<< ", particle count: " << goal_angles_to_particle_count[pair.first] << endl;
	}
	return oss.str();
}

// IS-DESPOT calls this method during Search()
// doesn't change the particle set iff all particles have equal weight
vector<State*> ParticleBelief::Sample(int num, vector<State*> particles, const DSPOMDP* model) 
{
	if(!Globals::config.NO_IMPORTANCE_SAMPLING) {
		double unit = 1.0 / num;
		double mass = Random::RANDOM.NextDouble(0, unit);
		int pos = 0;

		// doesn't change importance weight
		vector<double> importance_weight = model->ImportanceWeight(particles);

		// importance weight of first particle
		double cur = importance_weight[0];
		double total_weight=0;

		vector<State*> sample;
		for (int i = 0; i < num; i++) {
			// mass is random number between 0 and 1/#particles
			while (mass > cur) {
				pos++;
				// emulate modulo operation
				if (pos == particles.size())
					pos = 0;

				cur += importance_weight[pos];
			}
			// if importance weight is 0
			if (importance_weight[pos]==0) {
				printf("IS-DESPOT::[%s] Immportance weight of sampled particle must not be 0.\n", __PRETTY_FUNCTION__);
				exit(-1);
			}
			mass += unit;

			// sample particle whose importance weight caused cur >= mass
			State* particle = model->Copy(particles[pos]);
			// stays the same if the importance weight is uninformed, i.e. equals 1/#particles == unut
			particle->weight = unit*(particles[pos]->weight/importance_weight[pos]);
			total_weight += particle->weight;
			sample.push_back(particle);
		}

		for (int i=0; i<num; i++)
			sample[i]->weight /= total_weight;

		random_shuffle(sample.begin(), sample.end());
		return sample;
	}
	else if (Globals::config.AGENT != LEADER) {
		double unit = 1.0 / num;
		double mass = Random::RANDOM.NextDouble(0, unit);
		int pos = 0;
		double cur = particles[0]->weight;

		vector<State*> sample;
		for (int i = 0; i < num; i++) {
			while (mass > cur) {
				pos++;
				if (pos == particles.size())
					pos = 0;

				cur += particles[pos]->weight;
			}

			mass += unit;

			State* particle = model->Copy(particles[pos]);
			particle->weight = unit;
			sample.push_back(particle);
		}

		random_shuffle(sample.begin(), sample.end());
		return sample;
	} else {
		if (particles.size() != num) {
			printf("IS-DESPOT::[%s] Invalid number of sampled particles: Expected %d, got %lu.\n", 
			__PRETTY_FUNCTION__, num, particles.size());
			exit(-1);
		} else {
			return particles;
		}
	}
}


vector<State*> ParticleBelief::Resample(int num, const vector<State*>& belief,
	const DSPOMDP* model, History history, int hstart) {
	double unit = 1.0 / num;
	double mass = Random::RANDOM.NextDouble(0, unit);
	int pos = 0;
	double cur = belief[0]->weight;

	double reward;
	OBS_TYPE obs;

	vector<State*> sample;
	int count = 0;
	double max_wgt = Globals::NEG_INFTY;
	int trial = 0;
	while (count < num && trial < 200 * num) {
		// Pick next particle
		while (mass > cur) {
			pos++;
			if (pos == belief.size())
				pos = 0;

			cur += belief[pos]->weight;
		}
		trial++;

		mass += unit;

		State* particle = model->Copy(belief[pos]);

		// Step through history
		double log_wgt = 0;
		for (int i = hstart; i < history.Size(); i++) {
			model->Step(*particle, Random::RANDOM.NextDouble(),
				history.Action(i), reward, obs);

			double prob = model->ObsProb(history.Observation(i), *particle,
				history.Action(i));
			if (prob > 0) {
				log_wgt += log(prob);
			} else {
				model->Free(particle);
				break;
			}
		}

		// Add to sample if survived
		if (particle->IsAllocated()) {
			count++;

			particle->weight = log_wgt;
			sample.push_back(particle);

			max_wgt = max(log_wgt, max_wgt);
		}

		// Remove particles with very small weights
		if (count == num) {
			for (int i = sample.size() - 1; i >= 0; i--)
				if (sample[i]->weight - max_wgt < log(1.0 / num)) {
					model->Free(sample[i]);
					sample.erase(sample.begin() + i);
					count--;
				}
		}
	}

	double total_weight = 0;
	for (int i = 0; i < sample.size(); i++) {
		sample[i]->weight = exp(sample[i]->weight - max_wgt);
		total_weight += sample[i]->weight;
	}
	for (int i = 0; i < sample.size(); i++) {
		sample[i]->weight = sample[i]->weight / total_weight;
	}

	logb << "[Belief::Resample] Resampled " << sample.size() << " particles" << endl;
	for (int i = 0; i < sample.size(); i++) {
		logb << " " << i << " = " << *sample[i] << endl;
	}

	return sample;
}

vector<State*> ParticleBelief::Resample(int num, const DSPOMDP* model,
	const StateIndexer* indexer, ACT_TYPE action, OBS_TYPE obs) {
	if (indexer == NULL) {
		std::cerr << "[Belief::Resample] indexer cannot be null" << endl;
		exit(1);
	}

	vector<State*> sample;

	for (int s = 0; s < indexer->NumStates(); s++) {
		const State* state = indexer->GetState(s);
		double prob = model->ObsProb(obs, *state, action);
		if (prob > 0) {
			State* particle = model->Copy(state);
			particle->weight = prob;
			sample.push_back(particle);
		}
	}

	return sample;
}

vector<State*> ParticleBelief::Resample(int num, const Belief& belief, History history,
	int hstart) {
	double reward;
	OBS_TYPE obs;

	vector<State*> sample;
	int count = 0;
	int pos = 0;
	double max_wgt = Globals::NEG_INFTY;
	vector<State*> particles;
	int trial = 0;
	while (count < num || trial < 200 * num) {
		// Pick next particle
		if (pos == particles.size()) {
			particles = belief.Sample(num);
			pos = 0;
		}
		State* particle = particles[pos];

		trial++;

		// Step through history
		double log_wgt = 0;
		for (int i = hstart; i < history.Size(); i++) {
			belief.model_->Step(*particle, Random::RANDOM.NextDouble(),
				history.Action(i), reward, obs);

			double prob = belief.model_->ObsProb(history.Observation(i),
				*particle, history.Action(i));
			if (prob > 0) {
				log_wgt += log(prob);
			} else {
				belief.model_->Free(particle);
				break;
			}
		}

		// Add to sample if survived
		if (particle->IsAllocated()) {
			particle->weight = log_wgt;
			sample.push_back(particle);

			max_wgt = max(log_wgt, max_wgt);
			count++;
		}

		// Remove particles with very small weights
		if (count == num) {
			for (int i = sample.size() - 1; i >= 0; i--) {
				if (sample[i]->weight - max_wgt < log(1.0 / num)) {
					belief.model_->Free(sample[i]);
					sample.erase(sample.begin() + i);
					count--;
				}
			}
		}
		pos++;
	}

	// Free unused particles
	for (int i = pos; i < particles.size(); i++)
		belief.model_->Free(particles[i]);

	double total_weight = 0;
	for (int i = 0; i < sample.size(); i++) {
		sample[i]->weight = exp(sample[i]->weight - max_wgt);
		total_weight += sample[i]->weight;
	}
	for (int i = 0; i < sample.size(); i++) {
		sample[i]->weight = sample[i]->weight / total_weight;
	}

	logb << "[Belief::Resample] Resampled " << sample.size() << " particles" << endl;
	for (int i = 0; i < sample.size(); i++) {
		logb << " " << i << " = " << *sample[i] << endl;
	}

	return sample;
}

} // namespace despot


