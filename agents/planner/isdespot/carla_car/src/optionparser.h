#pragma once

#include <sys/stat.h>
#include <string.h>
#include <iostream>

#include <despot/util/optionparser.h>
#include <despot/util/logging.h>
#include <despot/core/globals.h>
#include "despot/util/util.h"

#include "math_utils.h"


class OptionParser { 
    private:
        inline static const char* agents_as_string = "IS-DESPOT, HyLEAP, HyLEAR, HyPLAN, LEADER";
        inline static const char* log_levels_as_string  = 
            "CONNECTOR, SIMULATOR, DESPOT, POMDP, WORLD, BELIEF, IMPORTANCE_SAMPLING";
        inline static despot::option::Descriptor* usage;

        inline static bool parse_bool(std::string bool_string) 
        {
            if (bool_string.compare("true") == 0) return true;
            else if (bool_string.compare("false") == 0) return false;
            else {
                printf("IS-DESPOT::[SETUP] Invalid bool value received: Expected either 'true' or 'false', got %s.\n", bool_string.c_str());
                exit(-1);
            }
        }

    public:
        enum OptionType {DISABLE, ENABLE, SPECIFY, OTHER};
        enum OptionIndex {
            // misc
            HELP,
            PORT,
            SCENARIO,
            START_TIMESTAMP,
            // logging
            LOGGING,
            REDIRECT_OUTPUT,
            OUTPUT_DIRECTORY,
            TRACKING,
            // agent-specific
            AGENT,
            DECOUPLE,
            HACKY,
            NO_VERTICAL_PRUNING,
            ATTENTION_SAMPLING,
            // meta
            AGGRESSIVE_BELIEF_UPDATES,
            MINIMAL_NOISE,
            NO_IMPORTANCE_SAMPLING,
            NO_NORMALIZATION,
            TIMEOUT,
            TIME_PER_PLANNING_STEP,
            MAX_EPISODE_STEPS,
            NOISE,
            SEED,
            // belief tree construction
            MAX_SEARCH_DEPTH,
            DISCOUNT,
            PARTICLE_NUMBER,
            GAP_REDUCTION_RATE,
            MAX_POLICY_SIM_LEN,
            PRUNE
        };

        static despot::option::Descriptor* build_usage() 
        {

            const char* agents_string_piece_one = "\t--agent <{";
            const char* agents_string_piece_two = "}> \tAgent-specific code to execute (default: IS-DESPOT).";
            char* agent_help_text;
            agent_help_text = (char*) calloc(strlen(agents_string_piece_one) + 
                                             strlen(agents_as_string) + 
                                             strlen(agents_string_piece_two) + 1, sizeof(char));

            strcpy(agent_help_text, agents_string_piece_one); // copy string one into buffer
            strcat(agent_help_text, agents_as_string); // append string two to the result
            strcat(agent_help_text, agents_string_piece_two); // append string two to the result


            const char* logging_string_piece_one = "\t--logging <{";
            const char* logging_string_piece_two = "}> \tContent-specific logging (default: none).";
            char* logging_help_text;
            logging_help_text = (char*) calloc(strlen(logging_string_piece_one) + 
                                               strlen(log_levels_as_string) + 
                                               strlen(logging_string_piece_two) + 1, sizeof(char));

            strcpy(logging_help_text, logging_string_piece_one); // copy string one into buffer
            strcat(logging_help_text, log_levels_as_string); // append string two to the result
            strcat(logging_help_text, logging_string_piece_two); // append string two to the result

            // index, type, shortopt, longopt, function checking argument, help text
            static despot::option::Descriptor usage[] = {
                // misc
                { HELP, OTHER, "", "help", despot::option::Arg::None, "  \t--help \tPrint usage and exit." },
                { PORT, SPECIFY, "", "port", despot::option::Arg::Required, "  \t--port <PORT> \tTCP port to connect to." },
                { SCENARIO, SPECIFY, "", "scenario", despot::option::Arg::Required, "  \t--scenario <SCENARIO> \tCARLA-CTS2 benchmark scenario(s) over which the agent is evaluated." },
                { START_TIMESTAMP, SPECIFY, "", "start_timestamp", despot::option::Arg::Required, "  \t--start_timestamp <TIMESTAMP> \tStarting time of the Python script calling IS-DSPOT." },
                // logging
                { LOGGING, ENABLE, "", "logging", despot::option::Arg::Optional, logging_help_text},
                { OUTPUT_DIRECTORY, SPECIFY, "", "output_directory", despot::option::Arg::Required, "\t--output_directory <PATH> \tSaves all generated files in specified path (default: CWD)."},                
                { REDIRECT_OUTPUT, ENABLE, "", "redirect_output_to", despot::option::Arg::Optional, "\t--redirect_output_to <FILENAME> \tRedirects all output generated to specified file (default: false)."},
                { TRACKING, ENABLE, "", "track_planning_effort", despot::option::Arg::Optional, "\t--track_planning_effort <{true, false}> \tLogs several metrics that measure planning effort (default: false)." },
                // agent-specific
                { AGENT, SPECIFY, "", "agent", despot::option::Arg::Required, agent_help_text},
                { DECOUPLE, ENABLE, "", "decouple", despot::option::Arg::Optional, "\t--decouple <{true, false}> \tDoesn't use DRL agent's belief state value estimate as upper bound during HyLEAP/HyPLAN training (default: false)." },
                { HACKY, ENABLE, "", "hacky", despot::option::Arg::Optional, "  \t--hacky <{true, false}> \tUses Florian's hacky IS-DESPOT variant for HyLEAP/HyPLAN (default: false)." },
                { NO_VERTICAL_PRUNING, ENABLE, "", "no_vertical_pruning", despot::option::Arg::Optional, "\t--no_vertical_pruning <{true, false}> \tDisables vertical pruning for HyPLAN, e.g. during training (default: false)." },
                { ATTENTION_SAMPLING, ENABLE, "", "attention_sampling", despot::option::Arg::Optional, "\t--attention_sampling <{true, false}> \tSamples pedestrian goal directions from attention distribution generated by LEADER (default: false)." },
                // meta
                { AGGRESSIVE_BELIEF_UPDATES, ENABLE, "", "aggressive_belief_updates", despot::option::Arg::Optional, "\t--aggressive_belief_updates <{true, false}> \tBelief is only influenced by current step (default: false)."},
                { MINIMAL_NOISE, ENABLE, "", "minimal_noise", despot::option::Arg::Optional, "\t--minimal_noise <{true, false}> \tRemoves as much (artificially injected) noise as possible during planning simulation (default: false)."},
                { NO_IMPORTANCE_SAMPLING, DISABLE, "", "no_importance_sampling", despot::option::Arg::Optional, "  \t--no_importance_sampling <{true, false}>  \tDo not use importance sampling (default: false)." },
                { NO_NORMALIZATION, DISABLE, "", "no_normalization", despot::option::Arg::Optional, "  \t--no_normalization <{true, false}> \tDisable normalization for importance distribution (default: false)." },
                { TIMEOUT, SPECIFY, "", "timeout", despot::option::Arg::Optional, "\t--timeout <arg>  \tSearch time per scene simulation step in seconds (default: 0.25)." },
                { TIME_PER_PLANNING_STEP, SPECIFY, "", "time_per_planning_step", despot::option::Arg::Optional, "\t--time_per_planning_step <arg>  \tTime between planning simulation steps during belief tree construction in seconds (default: 0.25)." },
                { MAX_EPISODE_STEPS, SPECIFY, "", "max_episode_steps", despot::option::Arg::Optional, "\t--max_episode_steps <arg>  \tMaximum number of steps for each simulated episode (default: 500)." }, 
                { NOISE, SPECIFY, "", "noise", despot::option::Arg::Optional, "\t--noise <arg>  \tNoise level for transition in POMDPX belief update (default: 0.1)." },
                { SEED, SPECIFY, "", "seed", despot::option::Arg::Optional, "\t--seed <arg>  \tRandom number seed (default: random)." },
                // belief tree construction
                { MAX_SEARCH_DEPTH, SPECIFY, "", "max_search_depth", despot::option::Arg::Optional, "\t--max_search_depth <arg>  \tMaximum search depth during belief tree construction (default: 20)." },
                { DISCOUNT, SPECIFY, "", "discount_factor", despot::option::Arg::Optional, "\t--discount_factor <arg>  \tFactor to discount future rewards (default 0.99)." },
                { PARTICLE_NUMBER, SPECIFY, "", "particle_number", despot::option::Arg::Optional, "\t--particle_number <arg>  \tNumber of particles used to approximate belief nodes (default: 500)." },
                { GAP_REDUCTION_RATE, SPECIFY, "", "gap_reduction_rate", despot::option::Arg::Optional, "\t--gap_reduction_rate <arg>  \tRequired gap reduction rate of each trial (default to 0.95)." },
                { MAX_POLICY_SIM_LEN, SPECIFY, "", "max_policy_simulation_length", despot::option::Arg::Optional, "\t--max_policy_simulation_length <arg>  \tNumber of steps to simulate the reactive controller at leaf nodes (default: 90)." },
                { PRUNE, SPECIFY, "", "pruning_constant", despot::option::Arg::Optional, "\t--pruning_constant <arg>  \tPruning constant for regularization (default: 0.0, i.e. no pruning)" },
                { 0, 0, 0, 0, 0, 0 }};
            return usage;
        }

        static void parse_arguments(int argc, char *argv[])
        {
            const char *program = (argc > 0) ? argv[0] : "IS-DESPOT";
            argc -= (argc > 0);
            argv += (argc > 0); // skip program name argv[0] if present

            OptionParser::usage = OptionParser::build_usage();
            despot::option::Stats stats(usage, argc, argv);
            despot::option::Option *options = new despot::option::Option[stats.options_max];
            despot::option::Option *buffer = new despot::option::Option[stats.buffer_max];
            despot::option::Parser parse(usage, argc, argv, options, buffer);

            if (options[OptionIndex::HELP]) {
                std::cout << "Usage: " << program << " [options]" << std::endl;
                despot::option::printUsage(std::cout, usage);
            }

            // misc
            if (options[PORT]) {
                int port = atoi(options[PORT].arg);
                if (port < 1024 || port > 65535) {
                    printf("IS-DESPOT::[SETUP] Invalid TCP port received: Expected 1024 <= PORT <= 65535, got %d\n", port);
                    exit(-1);
                }
                despot::Globals::config.PORT = port;
                printf("IS-DESPOT::[SETUP] Running on port %d.\n", despot::Globals::config.PORT);
            } else {
                printf("IS-DESPOT::[SETUP] No TCP port specified to connect to.\n");
                exit(-1);
            }

            if (options[SCENARIO]) {
                despot::Globals::config.SCENARIO = options[SCENARIO].arg;
                printf("IS-DESPOT::[SETUP] Running on scenario(s) %s.\n", despot::Globals::config.SCENARIO.c_str());
            } else {
                printf("IS-DESPOT::[SETUP] No scenario(s) specified: Required for preventing overwriting tracking files.\n");
                exit(-1);
            }

            if (options[START_TIMESTAMP]) {
                despot::Globals::config.START_TIMESTAMP = options[START_TIMESTAMP].arg;
                printf("IS-DESPOT::[SETUP] Starting time of script execution is %s.\n", despot::Globals::config.START_TIMESTAMP.c_str());
            } else {
                printf("IS-DESPOT::[SETUP] No start timestamp specified: Required for preventing overwriting tracking files.\n");
                exit(-1);
            }

            // logging
            despot::logging::initialize_settings(false);
            // prevents scientific notation and sets precision to 4 decimal numbers
            std::cout << std::setprecision(12) << std::fixed;
            // for all logging levels provided
            for (despot::option::Option* option = options[LOGGING]; option; option = option->next()) {
                std::string log_level = option->arg;
                bool match = false;
                for (const auto& map_entry: despot::logging::marker_to_scope) {
                    if (log_level.compare(map_entry.first) == 0) {
                        despot::logging::set_scope(map_entry.second, true);
                        printf("IS-DESPOT::[SETUP] Logging all %s related outputs.\n", map_entry.first.c_str());
                        match = true;
                        break;
                    }
                }
                if (!match) {
                    printf("IS-DESPOT::[SETUP] Invalid logging level specified: Expected one of {%s}, got %s.\n", 
                    log_levels_as_string, log_level.c_str());
                    exit(-1);
                }
            }

            if (options[OUTPUT_DIRECTORY]) {
                std::string output_directory = options[OUTPUT_DIRECTORY].arg;
                struct stat buffer;   
                // check for existence
                if (stat(output_directory.c_str(), &buffer) == 0 && (buffer.st_mode & S_IFDIR)) {
                    despot::Globals::config.OUTPUT_DIRECTORY = output_directory;
                    printf("IS-DESPOT::[SETUP] Logging output in directory '%s'.\n", output_directory.c_str());
                } else {
                    printf("IS-DESPOT::[SETUP] Invalid output directory %s. Does not exist.\n", output_directory.c_str());
                    exit(-1);
                }
            } else {
                std::string cwd = despot::get_cwd();
                printf("IS-DESPOT::[SETUP] No output directory specified. Logging into current working directory %s.\n", cwd.c_str());
                despot::Globals::config.OUTPUT_DIRECTORY = cwd;
            }

            // agent-specific
            if (options[AGENT]) {
                std::string agent_string = options[AGENT].arg;
                if (string_to_agent.find(agent_string) == string_to_agent.end()) {
                    printf("IS-DESPOT::[SETUP] Invalid agent specified: Expected one of {%s}, got %s.\n",
                    agents_as_string, agent_string.c_str());
                    exit(-1);
                }
                despot::Globals::config.AGENT = string_to_agent[agent_string];
            } else {
                printf("IS-DESPOT::[SETUP] No agent specified: Running IS-DESPOT.\n");
            }

            if (options[DECOUPLE]) {
                bool decouple = parse_bool(options[DECOUPLE].arg);
                if ((despot::Globals::config.AGENT != HyLEAP && despot::Globals::config.AGENT != HyPLAN) && decouple) {
                    printf("IS-DESPOT::[SETUP] Can only decouple training of HyLEAP or HyPLAN.\n");
                    exit(-1);
                }
                if (decouple) {
                     printf("IS-DESPOT::[SETUP] Decoupling HyLEAP/HyPLAN and IS-DESPOT for training.\n");
                }
                despot::Globals::config.DECOUPLE = decouple;
            }

            if (options[HACKY]) {
                bool hacky = parse_bool(options[HACKY].arg);
                if ((despot::Globals::config.AGENT != HyLEAP && despot::Globals::config.AGENT != HyPLAN) && hacky) {
                    printf("IS-DESPOT::[SETUP] Can only run hacky IS-DESPOT variant for HyLEAP or HyPLAN.\n");
                    exit(-1);
                }
                if (hacky && despot::Globals::config.DECOUPLE) {
                    printf("IS-DESPOT::[SETUP] Florian's hacky IS-DESPOT modifications for HyLEAP are already a partial decoupling.\n");
                    exit(-1);
                }
                if (hacky) {
                    printf("IS-DESPOT::[SETUP] Running Florian's hacky IS-DESPOT implementation.\n");
                }
                despot::Globals::config.HACKY = hacky;
            }

            if (options[NO_VERTICAL_PRUNING]) {
                bool no_vertical_pruning = parse_bool(options[NO_VERTICAL_PRUNING].arg);
                if ((despot::Globals::config.AGENT != HyPLAN) && no_vertical_pruning) {
                    printf("IS-DESPOT::[SETUP] Can only disable vertical pruning for HyPLAN.\n");
                    exit(-1);
                }
                if (no_vertical_pruning) {
                    printf("IS-DESPOT::[SETUP] Disabling HyPLAN's uncertainty-based vertical pruning.\n");
                }
                despot::Globals::config.NO_VERTICAL_PRUNING = no_vertical_pruning;
            }

            if (options[ATTENTION_SAMPLING]) {
                bool attention_sampling = parse_bool(options[ATTENTION_SAMPLING].arg);
                if ((despot::Globals::config.AGENT != LEADER) && attention_sampling) {
                    printf("IS-DESPOT::[SETUP] Can only enable attention distribution sampling for LEADER.\n");
                    exit(-1);
                }
                if (attention_sampling) {
                    printf("IS-DESPOT::[SETUP] Sampling pedestrian goal directions from attention distribution generated by LEADER.\n");
                }
                despot::Globals::config.ATTENTION_SAMPLING = attention_sampling;
            }

            // has to be parsed after agent-specific arguments, because tracking depends on the above
            if (options[TRACKING]) {
                despot::Globals::config.TRACKING = parse_bool(options[TRACKING].arg);
                if (despot::Globals::config.TRACKING) {
                    printf("IS-DESPOT::[SETUP] Tracking despot internal performance metrics. This will increase runtime consumption.\n");
                    despot::logging::initialize_tracking();
                }
            }

            // meta 
            if (options[AGGRESSIVE_BELIEF_UPDATES]) {
                despot::Globals::config.AGGRESSIVE_BELIEF_UPDATES = parse_bool(options[AGGRESSIVE_BELIEF_UPDATES].arg);
            } 
            if (options[MINIMAL_NOISE]) {
                despot::Globals::config.MINIMAL_NOISE = parse_bool(options[MINIMAL_NOISE].arg);
                    if (despot::Globals::config.MINIMAL_NOISE) {
                        ModelParams::NOISE_GOAL_ANGLE = M_PI * 0.005;
                        ModelParams::NOISE_ROBVEL = 0.0;
                }
            } 
            if (options[NO_IMPORTANCE_SAMPLING]) {
                despot::Globals::config.NO_IMPORTANCE_SAMPLING = parse_bool(options[NO_IMPORTANCE_SAMPLING].arg);
            } 
            if (options[NO_NORMALIZATION]) {
                despot::Globals::config.NO_NORMALIZATION = parse_bool(options[NO_NORMALIZATION].arg);
            } 
            if (options[TIMEOUT]) {
                double timeout = atof(options[TIMEOUT].arg);
                if (timeout <= 0.0) {
                    printf("IS-DESPOT::[SETUP] Invalid timeout specified: Expected TIMEOUT > 0.0, got %.4f.\n", timeout);
                    exit(-1);
                }
                despot::Globals::config.TIMEOUT = timeout;
            }

            if (options[TIME_PER_PLANNING_STEP]) {
                double time_per_planning_step = atof(options[TIME_PER_PLANNING_STEP].arg);
                if (time_per_planning_step <= 0.0) {
                    printf("IS-DESPOT::[SETUP] Invalid time per planning step specified: Expected TIME > 0.0, got %.4f.\n", 
                    time_per_planning_step);
                    exit(-1);
                }
                despot::Globals::config.TIME_PER_PLANNING_STEP = time_per_planning_step;
            }

            if (options[MAX_EPISODE_STEPS]) {
                int maximum_episode_steps = atoi(options[MAX_EPISODE_STEPS].arg);
                if (maximum_episode_steps <= 0) {
                    printf("IS-DESPOT::[SETUP] Invalid number of maximum episode steps: Epected STEPS > 0, got %d.\n", 
                    maximum_episode_steps);
                    exit(-1);
                }
                despot::Globals::config.MAX_EPISODE_STEPS = maximum_episode_steps;
            }

            if (options[NOISE]) {
                double noise = atof(options[NOISE].arg);
                if (noise < 0.0 || noise > 1.0) {
                    printf("IS-DESPOT::[SETUP] Invalid noise level: Expected 0.0 <= NOISE <= 1.0, got %.4f.\n", noise);
                    exit(-1);
                }
                despot::Globals::config.NOISE = noise;
            }

            if (options[SEED]) {
                despot::Globals::config.SEED = atoi(options[SEED].arg);
                despot::Seeds::root_seed(despot::Globals::config.SEED);
                despot::Random::RANDOM = despot::Random(despot::Globals::config.SEED);
            }

            // planning
            if (options[MAX_SEARCH_DEPTH]) {
                int depth = atoi(options[MAX_SEARCH_DEPTH].arg);
                if (depth <= 0) {
                    printf("IS-DESPOT::[SETUP] Invalid maximum planning depth: Expected DEPTH > 0, got %d.\n", depth);
                    exit(-1);
                }
                despot::Globals::config.MAX_SEARCH_DEPTH = depth;
            }

            if (options[DISCOUNT]) {
                double discount = atof(options[DISCOUNT].arg);
                if (discount < 0.0 || discount > 1.0) {
                    printf("IS-DESPOT::[SETUP] Invalid discount factor: Expected 0.0 <= DISCOUNT <= 1.0, got %.4f.\n", discount);
                    exit(-1);
                }
                despot::Globals::config.DISCOUNT = discount;
            }
                
            if (options[PARTICLE_NUMBER]) {
                int particle_number = atoi(options[PARTICLE_NUMBER].arg);
                if (particle_number <= 0) {
                    printf("IS-DESPOT::[SETUP] Invalid number of particles: Expected NUMBER > 0, got %d.\n", particle_number);
                    exit(-1);
                }
                despot::Globals::config.PARTICLE_NUMBER = particle_number;
            }

            if (options[GAP_REDUCTION_RATE]) {
                double gap = atof(options[GAP_REDUCTION_RATE].arg);
                if (gap < 0.0 || gap > 1.0) {
                    printf("IS-DESPOT::[SETUP] Invalid gap constant: Expected 0.0 <= GAP_REDUCTION_RATE <= 1.0, got %.4f.\n", gap);
                    exit(-1);
                }
                despot::Globals::config.GAP_REDUCTION_RATE = gap;
            }

            if (options[MAX_POLICY_SIM_LEN]) {
                int max_policy_sim_len = atoi(options[MAX_POLICY_SIM_LEN].arg);
                if (max_policy_sim_len < 0) {
                    printf("IS-DESPOT::[SETUP] Invalid number of default policy simulation length: " 
                           "Expected LENGTH > 0, got %d.\n", max_policy_sim_len);
                    exit(-1);
                }
                despot::Globals::config.MAX_POLICY_SIM_LEN = max_policy_sim_len;
            }
        
            if (options[PRUNE]) {
                double pruning_constant = atof(options[PRUNE].arg);
                if (pruning_constant < 0.0 || pruning_constant > 1.0) {
                    printf("IS-DESPOT::[SETUP] Invalid pruning constant: Expected 0.0 <= PRUNE <= 1.0, got %.4f.\n", 
                           pruning_constant);
                    exit(-1);
                }
                despot::Globals::config.PRUNING_CONSTANT = pruning_constant;                
            }

            // parse this last such that previous error messages are printed to the console
            if (options[REDIRECT_OUTPUT]) {
                std::string redirected_output_path = despot::Globals::config.OUTPUT_DIRECTORY + "/" + options[REDIRECT_OUTPUT].arg;                
                printf("IS-DESPOT::[SETUP] Redirecting console output to '%s'.\n", redirected_output_path.c_str());
                freopen(redirected_output_path.c_str(), "w", stdout);
            } 
        }
};