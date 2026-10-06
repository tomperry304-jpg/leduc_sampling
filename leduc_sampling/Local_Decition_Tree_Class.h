#pragma once
#include "Matrix_Class.h"
#include <vector>
#include <cstddef>
#include <string>
#include <iostream>
#include <stdexcept>
#include <omp.h>  
#include <random>
#include <unordered_map>
#include <numeric>

float sigmoid(float z);
std::vector<float> gamstate_normalisation(float max_pot, float max_player_contribution, int ranks, int n_players, std::vector<float> gamestate_vec);


class Decition_Node {
private:
    std::vector<float> cards_;
    std::vector<std::string> history_;
    std::vector<std::string> round_history_;
    int round_;
    std::vector<float> player_contributions_;
    int n_players_;
    float pot_;
    std::vector<std::string> player_status_;
    int player_turn_;
    int last_agressor_;
    bool round_raised_;
    std::vector<float> gamestate_vec_;
    int round_actions_count_;
    int max_actions_per_round_;
    int player_traversing_;
    static inline const std::unordered_map<std::string, int> action_index_ = {
    {"fold", 0}, {"check", 1}, {"call", 2}, {"bet", 3}, {"raise", 4}
    };
public:
    Decition_Node(int last_agressor, bool round_raised, std::vector<float> cards, int n_players, std::vector<std::string> history, std::vector<std::string> round_history, int round, std::vector<float> player_contributions, float pot, std::vector<std::string> player_status, int player_turn, std::vector<float> gamestate_vec, int round_actions_count, int max_actions_per_round, int player_traversing)
        : cards_(cards), history_(history), round_history_(round_history), round_(round), n_players_(n_players), player_contributions_(player_contributions), pot_(pot), player_status_(player_status), player_turn_(player_turn), last_agressor_(last_agressor), round_raised_(round_raised), gamestate_vec_(gamestate_vec), round_actions_count_(round_actions_count), max_actions_per_round_(max_actions_per_round), player_traversing_(player_traversing){
        gamestate_vec_[2] = cards_[player_traversing_];
        gamestate_vec_[5 + player_traversing_] = 1;
    }

    ~Decition_Node() {}

    //accecability funcitons
    int get_round() const { return round_; }
    int active_players() const { return std::count(player_status_.begin(), player_status_.end(), "playing"); }
    bool current_player_active() const { return player_status_[player_turn_] == "playing"; }

    bool checking_check(int n_seats) {
        bool check = true;
        for (int i = 2; i <= n_seats; i++) {
            if (round_history_[round_history_.size() - i] == "check") {
            }
            else {
                check = false;
                return check;
            }
        }
        return check;
    }

    int n_seats_() {
        return std::count(player_status_.begin(), player_status_.end(), "playing");
    }

    std::vector<std::string> legal_actions() {




        if (round_history_.empty()) {
            return { "check","bet" };
        }
        int n_seats = n_seats_();
        if (n_seats == 1) {
            return {};
        }

        const std::string& last = round_history_.back();
        if (last == "check") {
            if (round_history_.size() >= n_seats && checking_check(n_seats)) return {};
            return { "check","bet" };
        }

        if (last == "bet") {
            return { "fold","call","raise" };
        }
        if (last == "raise") {
            return { "fold","call" };
        }
        if (last == "fold" || last == "call") {
            if (player_turn_ == last_agressor_) {
                return{};
            }
            else {
                return { "fold","call" };
            }
        };
        return {};
    }

    float wining_card() {
        float public_card = cards_.back();
        float highest_card = -1e9;
        for (int i = 0; i < n_players_; i++) {
            if (cards_[i] == public_card and player_status_[i] == "playing") {
                return public_card;
            }
            else {
                if (cards_[i] > highest_card and player_status_[i] == "playing") {
                    highest_card = cards_[i];
                }
            }
        }
        return highest_card;
    }


    std::vector<float> terminal_returns() {
        std::vector<float> returns(player_contributions_.size(), 0.0f);
        int n_seats = std::count(player_status_.begin(), player_status_.end(), "playing");
        if (n_seats == 1) {
            auto it = std::find(player_status_.begin(), player_status_.end(), "playing");
            int winner_idx = std::distance(player_status_.begin(), it);
            for (int i = 0; i < n_players_; i++) {
                if (i == winner_idx) {
                    returns[i] = pot_ - player_contributions_[i];
                }
                else {
                    returns[i] = -1 * player_contributions_[i];
                }
            }
        }
        else {
            float best_card = wining_card();

            std::vector<int> winner_i;
            for (int i = 0; i < n_players_; i++) {
                returns[i] = -1 * player_contributions_[i];
                if (cards_[i] == best_card && player_status_[i] == "playing") {
                    winner_i.push_back(i);
                }
            }
            for (int i = 0; i < winner_i.size(); i++) {
                returns[winner_i[i]] += pot_ / winner_i.size();
            }
        }
        return returns;
    }


    Decition_Node apply_action(std::string action) {
        //cards_, history_, round_history_, round_, n_players_, player_contributions_, pot_, player_status_, player_turn_, last_agressor_, round_raised_;
        std::vector<float> new_gamestate_vec_ = gamestate_vec_;
        new_gamestate_vec_[5 + n_players_ + player_turn_] = 0;
        int action_i = 1;
        float new_pot_ = pot_;
        std::vector<float> new_player_contributions_ = player_contributions_;
        std::vector<std::string> new_player_status_ = player_status_;
        std::vector<std::string> new_history_ = history_;
        std::vector<std::string> new_round_history_ = round_history_;
        int new_last_agressor_ = last_agressor_;
        bool new_round_raised_ = round_raised_;
        int new_player_turn_ = player_turn_;
        if (action == "raise" || action == "call" || action == "bet") {
            //walk back untill hit
            bool hit = false;
            int i = 1;
            while (hit == false) {
                if (new_player_status_[((player_turn_ - i) % n_players_ + n_players_) % n_players_] == "playing") {
                    hit = true;
                }
                else if (i > n_players_) throw std::runtime_error("no active player found");
                else {
                    i++;
                }
            }
            int I = ((player_turn_ - i) % n_players_ + n_players_) % n_players_;
            new_pot_ -= new_player_contributions_[player_turn_];
            if (action == "raise") {
                new_player_contributions_[player_turn_] = new_player_contributions_[I] + 2 * round_;
                new_last_agressor_ = player_turn_;
                new_round_raised_ = true;
                action_i = 4;
            }
            else if (action == "bet") {
                new_player_contributions_[player_turn_] = new_player_contributions_[I] + 2 * round_;
                new_last_agressor_ = player_turn_;
                action_i = 3;
            }
            else {
                new_player_contributions_[player_turn_] = new_player_contributions_[I];
                action_i = 2;
            }
            new_pot_ += new_player_contributions_[player_turn_];
            new_gamestate_vec_[1] = new_pot_;
            new_gamestate_vec_[5 + n_players_ * 2 + player_turn_] = new_player_contributions_[player_turn_];
        }
        else if (action == "fold") {
            new_player_status_[player_turn_] = "folded";
            new_gamestate_vec_[5 + n_players_ * 3 + player_turn_] = 0;
            action_i = 0;
        }
        new_history_.push_back(action);
        new_round_history_.push_back(action);
        new_gamestate_vec_[5 + n_players_ * 4 + (round_-1) * max_actions_per_round_* (n_players_ + 5) + round_actions_count_ * (n_players_ + 5) + player_turn_] = 1;
        new_gamestate_vec_[5 + n_players_ * 4 + (round_-1) * max_actions_per_round_* (n_players_ + 5) + round_actions_count_ * (n_players_ + 5) + n_players_ + action_i] = 1;
        //walk back untill hit
        bool hit = false;
        int i = 1;
        while (hit == false) {
            if (new_player_status_[(player_turn_ + i) % n_players_] == "playing") {
                hit = true;
            }
            else if (i > n_players_) throw std::runtime_error("no active player found");
            else {
                i++;
            }
        }
        new_player_turn_ = (player_turn_ + i) % n_players_;
        new_gamestate_vec_[5 + n_players_ + new_player_turn_] = 1;
        int new_round_actions_count = round_actions_count_ + 1;
        return Decition_Node(new_last_agressor_, new_round_raised_, cards_, n_players_, new_history_, new_round_history_, round_, new_player_contributions_, new_pot_, new_player_status_, new_player_turn_, new_gamestate_vec_, new_round_actions_count, max_actions_per_round_, player_traversing_);
    }


    Decition_Node new_round() {
        std::vector<float> new_gamestate_vec_ = gamestate_vec_;
        //cards_, history_, round_history_, round_, n_players_, player_contributions_, pot_, player_status_, player_turn_, last_agressor_, round_raised_;
        std::vector<std::string> new_round_history_ = {};
        bool new_round_raised_ = false;
        int new_player_turn_;
        int new_round_ = 2;

        bool hit = false;
        int i = 0;
        while (hit == false) {
            if (player_status_[i % n_players_] == "playing") {
                hit = true;
            }
            else if (i > n_players_) throw std::runtime_error("no active player found");
            else {
                i++;
            }
        }
        new_player_turn_ = i % n_players_;
        new_gamestate_vec_[0] = 1;
        float public_card = cards_[n_players_];
        new_gamestate_vec_[3] = public_card+1;
        if (public_card == cards_[player_traversing_]) {
            new_gamestate_vec_[4] = 1;
        }
        int new_round_actions_count = 0;
        return Decition_Node(last_agressor_, new_round_raised_, cards_, n_players_, history_, new_round_history_, new_round_, player_contributions_, pot_, player_status_, new_player_turn_, new_gamestate_vec_, new_round_actions_count, max_actions_per_round_, player_traversing_);
    }




    std::string get_key() {
        std::string key = std::to_string((int)cards_[player_turn_]);
        if (round_ >= 2) key += "|Public" + std::to_string((int)cards_[n_players_]);
        for (std::string& a : history_) key += "|" + a;
        return key;
    }

    std::string get_prey_key() {
        std::string key = std::to_string((int)cards_[player_turn_]);
        key += "|Public" + std::to_string((int)cards_[n_players_]);
        for (std::string& a : history_) key += "|" + a;
        return key;
    }

    

    std::vector<float> recursion(std::unordered_map<std::string, std::vector<float>>& regrets, std::unordered_map<std::string, std::vector<float>>& strategy, std::vector<float> reach, int player_traverser, std::mt19937& rng, std::vector <std::vector<float>>& gamstate_training, std::vector <std::vector<float>>& regret_training, std::vector<float>& chance_training, float chance) {
        std::vector<std::string> actions = legal_actions();
        if (actions.empty()) {
            if (n_seats_() != 1 && round_ == 1) {
                Decition_Node child = new_round();
                return child.recursion(regrets, strategy, reach, player_traverser, rng, gamstate_training, regret_training, chance_training, chance);
            }
            else {
                return terminal_returns();
            }
        }
        else {
            std::vector<float> node_value(n_players_, 0.0f);
            std::vector< std::vector<float>> exp_action_i;
            std::string key = get_key();
            auto it = regrets.find(key);
            if (it == regrets.end()) {
                regrets[key] = std::vector<float>(actions.size(), 0.0f);
                strategy[key] = std::vector<float>(actions.size(), 0.0f);
            }
            std::vector<float> regret = regrets[key];
            float total_regret = 0;
            for (int i = 0; i < regret.size(); i++) {
                if (regret[i] <= 0) {
                    regret[i] = 0;
                }
                total_regret += regret[i];
            }
            if (player_turn_ == player_traverser) {
                std::vector<float> regrets_training_node(5, 0);
                std::vector<float> gamstate_training_node = gamestate_vec_;
                for (int i = 0; i < actions.size(); i++) {
                    Decition_Node child = apply_action(actions[i]);

                    float p = 0;
                    if (total_regret == 0) {
                        p = 1.0f / regret.size();
                    }
                    else {
                        p = regret[i] / total_regret;
                    }
                    std::vector<float> new_reach = reach;
                    new_reach[player_turn_] *= p;


                    std::vector<float> exp = child.recursion(regrets, strategy, new_reach, player_traverser, rng, gamstate_training, regret_training, chance_training, chance);
                    exp_action_i.push_back(exp);
                    for (size_t i = 0; i < node_value.size(); i++) {
                        node_value[i] += p * exp[i];
                    }

                    strategy[key][i] += chance * reach[player_turn_] * p;
                }
                for (int i = 0; i < actions.size(); i++) {
                    regrets[key][i] += chance * (exp_action_i[i][player_turn_] - node_value[player_turn_]);
                    int action_i = action_index_.at(actions[i]);
                    regrets_training_node[action_i] = exp_action_i[i][player_turn_] - node_value[player_turn_];
                }
                gamstate_training.push_back(gamstate_training_node);
                regret_training.push_back(regrets_training_node);
                chance_training.push_back(chance);
            }
            else {
                std::vector<float> p_dist;
                p_dist.resize(actions.size());
                if (total_regret == 0) {
                    p_dist = std::vector < float>(regret.size(), 1.0f / regret.size());
                }
                else {
                    for (int i = 0; i < actions.size(); i++) {
                        p_dist[i] = regret[i] / total_regret;
                    }
                }
                std::discrete_distribution<int> dist(p_dist.begin(), p_dist.end());
                int chosen = dist(rng);
                Decition_Node child = apply_action(actions[chosen]);
                node_value = child.recursion(regrets, strategy, reach, player_traverser, rng, gamstate_training, regret_training, chance_training, chance);
            }


            return node_value;
        }
    }
    //####################################################################################################
    //####################################################################################################
    //####################################################################################################
    std::vector<float> player_centric_gamstate(int player_turn) {
        std::vector<float> player_gamestate_vec = gamestate_vec_;
        player_gamestate_vec[2] = cards_[player_turn];
        if (player_gamestate_vec[3]-1 == player_gamestate_vec[2]) {
            player_gamestate_vec[4] = 1;
        }
        else {
            player_gamestate_vec[4] = 0;
        }
        return player_gamestate_vec;
    }

    std::vector<float> forward_pass(std::vector<float> input, std::vector <Matrix>& WN, int hidden_layer_size, int n_hidden_layers, float max_pot){
        std::vector<float> l1_w1 = input * WN[0];
        std::vector<float> a1(hidden_layer_size + 1);
        std::transform(l1_w1.begin(), l1_w1.end(), a1.begin(), sigmoid);
        a1[hidden_layer_size] = 1;
        std::vector<std::vector<float>> an;
        an.push_back(a1);
        for (int k = 0; k < n_hidden_layers - 1; k++) {
            std::vector<float> lk_wk = an[k] * WN[k + 1];
            std::vector<float> ak(hidden_layer_size + 1);
            std::transform(lk_wk.begin(), lk_wk.end(), ak.begin(), sigmoid);
            ak[hidden_layer_size] = 1;
            an.push_back(ak);
        }
        std::vector<float> lfinal_wfinal = an.back() * WN[n_hidden_layers];
        std::vector<float> afinal = lfinal_wfinal;
        for (int i = 0; i < 5; i++) {
            afinal[i] *= max_pot;
        }
        return afinal;
    }

    std::vector<float> stratag_forward_pass(std::vector<float> input, std::vector <Matrix>& WN_stratagy, int hidden_layer_size, int n_hidden_layers) {
        std::vector<float> l1_w1 = input * WN_stratagy[0];
        std::vector<float> a1(hidden_layer_size + 1);
        std::transform(l1_w1.begin(), l1_w1.end(), a1.begin(), sigmoid);
        a1[hidden_layer_size] = 1;
        std::vector<std::vector<float>> an;
        an.push_back(a1);
        for (int k = 0; k < n_hidden_layers - 1; k++) {
            std::vector<float> lk_wk = an[k] * WN_stratagy[k + 1];
            std::vector<float> ak(hidden_layer_size + 1);
            std::transform(lk_wk.begin(), lk_wk.end(), ak.begin(), sigmoid);
            ak[hidden_layer_size] = 1;
            an.push_back(ak);
        }
        std::vector<float> lfinal_wfinal = an.back() * WN_stratagy[n_hidden_layers];
        std::vector<float> afinal = lfinal_wfinal;
        return afinal;
    }

    std::vector<float> nerual_network_recursion(std::unordered_map<std::string, std::vector<float>>& regrets, std::unordered_map<std::string, std::vector<float>>& strategy, std::vector<float> reach, int player_traverser, std::mt19937& rng, std::vector <std::vector<float>>& gamstate_training, std::vector <std::vector<float>>& regret_training, std::vector<float>& chance_training, float chance, float max_pot, float max_player_contribution, int ranks, std::vector<Matrix>& WN,int hidden_layer_size, int n_hidden_layers, std::vector <std::vector<float>>& stratagy_training, float max_capacity, float& N_samples_colected) {
        std::vector<std::string> actions = legal_actions();
        if (actions.empty()) {
            if (n_seats_() != 1 && round_ == 1) {
                Decition_Node child = new_round();
                return child.nerual_network_recursion(regrets, strategy, reach, player_traverser, rng, gamstate_training, regret_training, chance_training, chance, max_pot, max_player_contribution, ranks, WN, hidden_layer_size, n_hidden_layers, stratagy_training, max_capacity, N_samples_colected);
            }
            else {
                return terminal_returns();
            }
        }
        else {
            std::vector<float> node_value(n_players_, 0.0f);
            std::vector< std::vector<float>> exp_action_i;
            std::string key = get_key();
            auto it = strategy.find(key);
            if (it == strategy.end()) {
                strategy[key] = std::vector<float>(actions.size(), 0.0f);
            }
            std::vector<float> input;
            if (player_turn_ == player_traverser) {
                input = gamstate_normalisation(max_pot, max_player_contribution, ranks, n_players_, gamestate_vec_);
                input.push_back(1.0f);
            }
            else {
                std::vector<float> player_gamestate_vec = player_centric_gamstate(player_turn_);
                input = gamstate_normalisation(max_pot, max_player_contribution, ranks, n_players_, player_gamestate_vec);
                input.push_back(1.0f);
            }
            std::vector<float> raw = forward_pass(input, WN, hidden_layer_size, n_hidden_layers, max_pot);
            std::vector<float> regret(actions.size());
            for (size_t i = 0; i < actions.size(); i++) {
                float r = raw[action_index_.at(actions[i])];
                regret[i] = (r > 0.0f) ? r : 0.0f;
            }

            float total_regret = 0;
            for (int i = 0; i < regret.size(); i++) {
                if (regret[i] <= 0) {
                    regret[i] = 0;
                }
                total_regret += regret[i];
            }
            if (player_turn_ == player_traverser) {
                
                std::vector<float> regrets_training_node(5, 0);
                std::vector<float> stratagy_training_node(5, 0);
                std::vector<float> gamstate_training_node = gamestate_vec_;
                for (int i = 0; i < actions.size(); i++) {
                    Decition_Node child = apply_action(actions[i]);

                    float p = 0;
                    if (total_regret == 0) {
                        p = 1.0f / regret.size();
                    }
                    else {
                        p = regret[i] / total_regret;
                    }
                    std::vector<float> new_reach = reach;
                    new_reach[player_turn_] *= p;

                    

                    std::vector<float> exp = child.nerual_network_recursion(regrets, strategy, new_reach, player_traverser, rng, gamstate_training, regret_training, chance_training, chance, max_pot, max_player_contribution, ranks, WN, hidden_layer_size, n_hidden_layers, stratagy_training, max_capacity, N_samples_colected);
                    exp_action_i.push_back(exp);
                    for (size_t i = 0; i < node_value.size(); i++) {
                        node_value[i] += p * exp[i];
                    }


                    strategy[key][i] += chance * reach[player_turn_] * p;
                    int action_i = action_index_.at(actions[i]);
                    stratagy_training_node[action_i] = reach[player_turn_] * p;

                }
                for (int i = 0; i < actions.size(); i++) {
                    int action_i = action_index_.at(actions[i]);
                    regrets_training_node[action_i] = exp_action_i[i][player_turn_] - node_value[player_turn_];
                }
                N_samples_colected += 1;
                if (N_samples_colected <= max_capacity) {
                    gamstate_training.push_back(gamstate_training_node);
                    regret_training.push_back(regrets_training_node);
                    stratagy_training.push_back(stratagy_training_node);
                    chance_training.push_back(chance);
                }
                else {
                    std::vector<float> p_dist0;
                    float p = max_capacity / N_samples_colected;
                    p_dist0 = { p, 1 - p };
                    std::discrete_distribution<int> dist(p_dist0.begin(), p_dist0.end());
                    int chosen0 = dist(rng);
                    if (chosen0 == 0) {
                        std::uniform_int_distribution<int> dist(0, max_capacity - 1);
                        int I = dist(rng);
                        gamstate_training[I] = gamstate_training_node;
                        regret_training[I] = regrets_training_node;
                        stratagy_training[I] = stratagy_training_node;
                        chance_training[I] = chance;
                    }
                }
            }
            else {
                std::vector<float> p_dist;
                p_dist.resize(actions.size());
                if (total_regret == 0) {
                    p_dist = std::vector < float>(regret.size(), 1.0f / regret.size());
                }
                else {
                    for (int i = 0; i < actions.size(); i++) {
                        p_dist[i] = regret[i] / total_regret;
                    }
                }
                std::discrete_distribution<int> dist(p_dist.begin(), p_dist.end());
                int chosen = dist(rng);
                Decition_Node child = apply_action(actions[chosen]);
                node_value = child.nerual_network_recursion(regrets, strategy, reach, player_traverser, rng, gamstate_training, regret_training, chance_training, chance, max_pot, max_player_contribution, ranks, WN, hidden_layer_size, n_hidden_layers, stratagy_training, max_capacity, N_samples_colected);
            }


            return node_value;
        }
    }

    //####################################################################################################
    //####################################################################################################
    //####################################################################################################

    void final_stratagy_map_recursion(std::unordered_map<std::string, std::vector<float>>& regrets, std::unordered_map<std::string, std::vector<float>>& strategy, std::vector<float> reach, int player_traverser, std::mt19937& rng, std::vector <std::vector<float>>& gamstate_training, std::vector <std::vector<float>>& regret_training, std::vector<float>& chance_training, float chance, float max_pot, float max_player_contribution, int ranks, std::vector<Matrix>& WN_stratagy, int hidden_layer_size, int n_hidden_layers, std::vector <std::vector<float>>& stratagy_training, std::unordered_map<std::string, std::vector<std::vector<float>>>& network_stratagy_map) {
        std::vector<std::string> actions = legal_actions();
        if (actions.empty()) {
            if (n_seats_() != 1 && round_ == 1) {
                Decition_Node child = new_round();
                child.final_stratagy_map_recursion(regrets, strategy, reach, player_traverser, rng, gamstate_training, regret_training, chance_training, chance, max_pot, max_player_contribution, ranks, WN_stratagy, hidden_layer_size, n_hidden_layers, stratagy_training, network_stratagy_map);
            }
        }
        else {
            std::vector<float> node_value(n_players_, 0.0f);
            std::vector< std::vector<float>> exp_action_i;
            std::string key = get_key();
            auto it = strategy.find(key);
            if (it != strategy.end()) {

                std::vector<float> input;
                if (player_turn_ == player_traverser) {
                    input = gamstate_normalisation(max_pot, max_player_contribution, ranks, n_players_, gamestate_vec_);
                    input.push_back(1.0f);
                }
                else {
                    std::vector<float> player_gamestate_vec = player_centric_gamstate(player_turn_);
                    input = gamstate_normalisation(max_pot, max_player_contribution, ranks, n_players_, player_gamestate_vec);
                    input.push_back(1.0f);
                }
                std::vector<float> raw = stratag_forward_pass(input, WN_stratagy, hidden_layer_size, n_hidden_layers);
                std::vector<float> network_stratagy_node(5);
                std::vector<float> CFR_stratagy_node(5);
                for (size_t i = 0; i < actions.size(); i++) {
                    float s = raw[action_index_.at(actions[i])];
                    int action_i = action_index_.at(actions[i]);
                    network_stratagy_node[action_i] = (s > 0.0f) ? s : 0.0f;
                    CFR_stratagy_node[action_i] = strategy[key][i];
                }
                float network_strat_total = 0;
                for (int i = 0; i < 5; i++) {
                    network_strat_total += network_stratagy_node[i];
                }
                if (network_strat_total != 0) {
                    for (int i = 0; i < 5; i++) {
                        network_stratagy_node[i] /= network_strat_total;
                    }
                }
                else {
                    float I = actions.size();
                    for (size_t i = 0; i < actions.size(); i++) {
                        int action_i = action_index_.at(actions[i]);
                        network_stratagy_node[action_i] = 1 /I;

                    }
                }
                network_stratagy_map[key] = { CFR_stratagy_node ,network_stratagy_node };
            }
            for (int i = 0; i < actions.size(); i++) {
                Decition_Node child = apply_action(actions[i]);
                child.final_stratagy_map_recursion(regrets, strategy, reach, player_traverser, rng, gamstate_training, regret_training, chance_training, chance, max_pot, max_player_contribution, ranks, WN_stratagy, hidden_layer_size, n_hidden_layers, stratagy_training, network_stratagy_map);
            }            
        }
    }

    //####################################################################################################
    //####################################################################################################
    //####################################################################################################

    class predetor_hand_ {
    private:
        std::vector<float> deck_;
        std::vector<float> cards_;

        int n_players_;
        int predator_;
        float predator_card_;
        int unique_cards_;
        float chance_;
    public:
        predetor_hand_(std::vector<float> deck, std::vector<float> cards, int n_players, int predetor, int unique_cards, float predator_card, float chance)
            : deck_(deck), cards_(cards), n_players_(n_players), predator_(predetor), unique_cards_(unique_cards), predator_card_(predator_card), chance_(chance) {}
        ~predetor_hand_() {}

        std::vector<float> cards() {
            return cards_;
        }

        void branch(std::vector<std::vector<float>>& Possible_Cards, std::vector<float>& chance_vec) {
            if (cards_.size() == n_players_ + 1) {
                Possible_Cards.push_back(cards_);
                chance_vec.push_back(chance_);
            }
            else if (cards_.size() == predator_) {
                int n = std::count(deck_.begin(), deck_.end(), predator_card_);
                std::vector<float> deck = deck_;
                std::vector<float> cards = cards_;
                if (n > 0) {
                    std::vector<float> new_deck = deck;
                    auto it = std::find(new_deck.begin(), new_deck.end(), predator_card_);
                    if (it != new_deck.end()) {
                        new_deck.erase(it);
                    }
                    std::vector<float> new_cards = cards;
                    new_cards.push_back(predator_card_);
                    float new_chance = chance_ * (static_cast<float>(n) / deck_.size());
                    predetor_hand_ child(new_deck, new_cards, n_players_, predator_, unique_cards_, predator_card_, new_chance);
                    child.branch(Possible_Cards, chance_vec);
                }
            }
            else {
                for (float i = 0; i < unique_cards_; i++) {
                    int n = std::count(deck_.begin(), deck_.end(), i);
                    std::vector<float> deck = deck_;
                    std::vector<float> cards = cards_;
                    if (n > 0) {
                        std::vector<float> new_deck = deck;
                        auto it = std::find(new_deck.begin(), new_deck.end(), i);
                        if (it != new_deck.end()) {
                            new_deck.erase(it);
                        }
                        std::vector<float> new_cards = cards;
                        new_cards.push_back(i);
                        float new_chance = chance_ * (static_cast<float>(n) / deck_.size());
                        predetor_hand_ child(new_deck, new_cards, n_players_, predator_, unique_cards_, predator_card_, new_chance);
                        child.branch(Possible_Cards, chance_vec);
                    }
                }
            }
        }
    };




    static std::vector<std::vector<float>> possible_hands_calc(std::vector<float> deck, std::vector<float> cards, int n_players, int predetor, int unique_cards, float predator_card, float chance) {
        std::vector< std::vector<float>> possible_cards{};
        std::vector<float> chance_vec{};
        predetor_hand_ root(deck, cards, n_players, predetor, unique_cards, predator_card, chance);
        root.branch(possible_cards, chance_vec);
        possible_cards.push_back(chance_vec);
        return possible_cards;
    }




    //#########################################################################################################
    //#########################################################################################################
    //#########################################################################################################

    std::vector<float> exploitiblity_calc(std::unordered_map<std::string, std::vector<float>>& strategy, int predator, int predetor_card, std::vector<std::vector<float>> possible_hands, std::vector<float> weight_vec, int ranks) {
        std::vector<std::string> actions = legal_actions();
        if (actions.empty()) {
            if (n_seats_() != 1 && round_ == 1) {
                Decition_Node child = new_round();
                return child.exploitiblity_calc(strategy, predator, predetor_card, possible_hands, weight_vec, ranks);
            }
            else {
                std::vector<float> node_value(n_players_, 0.0f);
                for (int j = 0; j < possible_hands.size(); j++) {
                    Decition_Node child(last_agressor_, round_raised_, possible_hands[j], n_players_, history_, round_history_, round_, player_contributions_, pot_, player_status_, player_turn_, gamestate_vec_,round_actions_count_,max_actions_per_round_, player_traversing_);
                    std::string key_for_combo_j = child.get_key();
                    std::vector<float> v = child.terminal_returns();
                    for (int k = 0; k < n_players_; k++) {
                        node_value[k] += v[k] * weight_vec[j];
                    }
                }
                return node_value;
            }
        }
        else {
            std::vector<float> node_value(n_players_, 0.0f);
            if (round_ == 1 or player_turn_ != predator) {

                std::vector< std::vector<float>> exp_action_i;
                std::vector<float> oponent_reach;
                std::string key = get_key();

                float max_return = -1e9;

                for (int i = 0; i < actions.size(); i++) {

                    if (player_turn_ == predator) {
                        Decition_Node child = apply_action(actions[i]);
                        std::vector<float> v = child.exploitiblity_calc(strategy, predator, predetor_card, possible_hands, weight_vec, ranks);
                        if (v[predator] > max_return) {
                            node_value = v;
                            max_return = v[predator];
                        }
                    }
                    else {
                        Decition_Node root = apply_action(actions[i]);
                        std::vector<float> new_weight_vec = weight_vec;
                        for (int j = 0; j < possible_hands.size(); j++) {
                            Decition_Node child(last_agressor_, round_raised_, possible_hands[j], n_players_, history_, round_history_, round_, player_contributions_, pot_, player_status_, player_turn_, gamestate_vec_, round_actions_count_, max_actions_per_round_, player_traversing_);
                            std::string key_for_combo_j = child.get_key();
                            new_weight_vec[j] *= strategy.at(key_for_combo_j)[i];
                        }
                        std::vector<float> v = root.exploitiblity_calc(strategy, predator, predetor_card, possible_hands, new_weight_vec, ranks);
                        for (int k = 0; k < n_players_; k++) {
                            node_value[k] += v[k];
                        }
                    }
                }
            }
            else {
                for (int l = 0; l < ranks; l++) {
                    if (l == cards_[n_players_]) {
                        std::vector< std::vector<float>> filterd_hands;
                        std::vector<float> filtder_weights;
                        std::vector<float> filterd_node_value(n_players_, 0.0f);
                        for (int j = 0; j < possible_hands.size(); j++) {
                            if (possible_hands[j].back() == l) {
                                filterd_hands.push_back(possible_hands[j]);
                                filtder_weights.push_back(weight_vec[j]);
                            }
                        }
                        float max_return = -1e9;
                        for (int i = 0; i < actions.size(); i++) {
                            Decition_Node child = apply_action(actions[i]);
                            std::vector<float> v = child.exploitiblity_calc(strategy, predator, predetor_card, filterd_hands, filtder_weights, ranks);
                            if (v[predator] > max_return) {
                                filterd_node_value = v;
                                max_return = v[predator];
                            }
                        }
                        for (int k = 0; k < n_players_; k++) {
                            node_value[k] += filterd_node_value[k];
                        }
                    }
                }
            }
            return node_value;
        }
    }
};