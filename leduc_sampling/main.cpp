#include "Matrix_Class.h"
#include "Local_Decition_Tree_Class.h"

#include <cmath>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <unordered_map>
#include <algorithm>







float tanh_derivative_from_activation(float a) {
    return 1 - a * a;
}

float sigmoid(float z) {
    return 1 / (1 + std::exp(-z));
}

float delta_sigmoid(float a) {
    return a - (a * a);
}

float loss_1(float a) {
    return std::log(a + 1e-12f);
}

float loss_2(float a) {
    return std::log(1 - a + 1e-12f);
}


//###########################################################################################################################
//###########################################################################################################################
//###########################################################################################################################
//###########################################################################################################################





std::vector<float> deck_maker(int n_players, int ranks_) {
    int j_loop = std::max(2, (n_players + ranks_) / ranks_);
    std::vector<float> deck = {};
    for (int i = 0; i < ranks_; i++) {
        for (int j = 0; j < j_loop; j++) {
            deck.push_back(i);
        }
    }
    return deck;
}

std::vector<float> gamstate_normalisation(float max_pot, float max_player_contribution, int ranks,int n_players, std::vector<float> gamestate_vec) {
    gamestate_vec[1] /= max_pot;
    gamestate_vec[2] = (gamestate_vec[2]+1)/ranks;
    gamestate_vec[3] /= ranks;
    for (int i = 0; i < n_players; i++) {
        gamestate_vec[5 + n_players * 2 + i] /= max_player_contribution;
    }
    return gamestate_vec;
}

std::vector<float> regret_normalisation(float max_pot, std::vector<float> regret_vec) {
    for (int i = 0; i < 5; i++) {
        regret_vec[i] /= max_pot;
    }
    return regret_vec;
}


class hand_ {
private:
    std::vector<float> deck_;
    std::vector<float> cards_;

    int n_players_;
    float p_;
    int ranks_;
public:
    hand_(std::vector<float> deck, std::vector<float> cards, int n_players, float p, int ranks)
        : deck_(deck), cards_(cards), n_players_(n_players), p_(p), ranks_(ranks) {}

    ~hand_() {}

    void branch(std::vector<hand_>& Cards_Delt) {
        if (cards_.size() == n_players_ + 1) {
            Cards_Delt.push_back(hand_(deck_, cards_, n_players_, p_, ranks_));
        }
        else {
            for (float i = 0; i < ranks_; i++) {
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
                    float new_p = p_ * (static_cast<float>(n) / deck_.size());
                    hand_ child(new_deck, new_cards, n_players_, new_p, ranks_);
                    child.branch(Cards_Delt);
                }
            }
        }
    }
    //accecability functions
    std::vector<float> cards() {
        return cards_;
    }
    float p() {
        return p_;
    }
};

void weight_update(std::vector<Matrix>& WN, std::vector<float> input, int hidden_layer_size, int n_hidden_layers, std::vector<float> output, float scale, std::vector<float>& loss_tracking, float n) {
    // forward pass
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


    //loss per image calc
    float a0 = 0;
    std::vector<float> loss;
    for (int k = 0; k < 5; k++) {
        a0 += (output[k] - afinal[k]) * (output[k] - afinal[k]);
        loss.push_back((afinal[k] - output[k]) * 2 * scale);
    }
    loss_tracking.push_back(a0 * scale);
    //

    //


    // backward pass (ussing cross entropy+sigmoid)
    std::vector<float> Deltafinal = loss;
    std::vector < std::vector<float>> Delta_k;
    Delta_k.push_back(Deltafinal);
    an.push_back(afinal);
    //Delta2 Calcs
    for (int k = 0; k < n_hidden_layers; k++) {

        Matrix Wk = WN[n_hidden_layers - k];
        std::vector<float> ak = an[an.size() - 2 - k];
        std::vector<float> Sig_Delta(ak.size());
        std::transform(ak.begin(), ak.end(), Sig_Delta.begin(), delta_sigmoid);
        std::vector<float> Deltak = matrix_vec_multiply_transposed(Delta_k[k],Wk);
        int S = Deltak.size();
        for (int l = 0; l < S; l++) {
            Deltak[l] *= Sig_Delta[l];
        }
        Delta_k.push_back(Deltak);
    }
    //grad W1 W2
    std::vector<std::vector<float>> feeding_activations;
    for (int k = n_hidden_layers - 1; k >= 0; k--) feeding_activations.push_back(an[k]);
    feeding_activations.push_back(input);

    std::vector<Matrix> grad_k;
    for (int k = 0; k < Delta_k.size(); k++) {
        grad_k.push_back(feeding_activations[k] * Delta_k[k]);
    }

    for (int k = 0; k < Delta_k.size(); k++) {
        WN[k].subtracing_scaled_matrix(grad_k[Delta_k.size() - 1 - k],n);
    }
};

int main() {
    std::cout << omp_get_max_threads();

    int n_players = 2;
    int B = 2;
    int max_actions_per_round = n_players + (n_players - 1) * B;
    int ant = 1;
    int b1 = 2;
    int b2 = 4;
    float max_player_contribution = ant + b1 * B + b2 * B;
    float max_pot = max_player_contribution * n_players;
    int n_actions = 5;
    std::vector<float> gamstate_vec(5 + n_players * 4 + max_actions_per_round * 2 * (n_actions + n_players), 0.0f);
    gamstate_vec[0] = 0;//round 1
    gamstate_vec[1] = (ant * n_players);
    gamstate_vec[5+n_players] = 1;//player 1 turn
    for (int i = 0; i < n_players; i++) {
        gamstate_vec[5 + 2*n_players + i] = ant;//inital contributions = ant
    }
    for (int i = 0; i < n_players; i++) {
        gamstate_vec[5 + 3*n_players + i] = 1;//all player playing
    }
    //round
    //pot
    //player-card
    //public-card
    //pair
    //player-traversing
    //player-turn
    //player-contribution
    //player-state
    //round1-history (player-turn,player-action)
    //round2-history (player-turn,player-action)

    int ranks = 3;
    std::vector<float> deck = deck_maker(n_players, ranks);
    std::vector<float> player_contributions(n_players, 1.0f);
    std::vector<std::string> player_status(n_players, "playing");
    std::vector<float> reach(n_players, 1.0f);
    std::vector<hand_> Cards_Delt;
    hand_ hand_root(deck, {}, n_players, 1.0f, ranks);
    hand_root.branch(Cards_Delt);

    std::unordered_map<std::string, std::vector<float>> regrets;
    std::unordered_map<std::string, std::vector<float>> strategy;
   

    std::vector<std::vector<float>> gamstate_training;
    std::vector<std::vector<float>> regret_training;
    std::vector<std::vector<float>> stratagy_training;
    std::vector<float> chance_training;

    std::mt19937 rng(42);

    int last_agressor = -1;
    bool round_raised = false;

    std::vector<std::string> history = {};
    std::vector<std::string> round_history = {};

    int player_turn = 0;
    float pot = n_players * 1.0f;
    int round = 1;
    int  round_actions_count = 0;

    int N = gamstate_vec.size();
    int gernerations = 10;
    int alpha = 2;
    int W_min = 128;
    int W_max = 1024;
    int hidden_layer_size = std::clamp(alpha * N, W_min, W_max);
    int L_min = 2;
    int n_hidden_layers = L_min + floor(log2(n_players));
    int final_layer_size = 5;

    int n_generations = 1000;
    int n_per_network_update = 10;

    float n0 = 0.1;
    float f = 0.1;
    float k = (1/f-1) / ((n_generations / n_per_network_update) * gernerations);
    int network_count = 0;
    //initalise network;
    Matrix Wfirst = Matrix::Random_Intaliser(N + 1, hidden_layer_size + 1);//+1 for biase
    std::vector<Matrix> WN;
    WN.push_back(Wfirst);
    for (int i = 0; i < n_hidden_layers - 1; i++) {
        WN.push_back(Matrix::Random_Intaliser(hidden_layer_size + 1, hidden_layer_size + 1));
    }
    Matrix Wlast = Matrix::Random_Intaliser(hidden_layer_size + 1, final_layer_size);//+1 for biase
    WN.push_back(Wlast);

    //stragy initalisation
    std::vector<Matrix> WN_stratagy = WN;

    std::vector<float> loss_tracking_regret;
    std::vector<float> loss_tracking_stratagy;
    std::vector<float> convergence_tracking;
    std::vector<float> convergence_tracking_stratagy;
    //######################################################
    //############################################################

    float max_capacity = 5000;

    float N_samples_colected = 0;
    
    for (int j = 0; j < n_generations; j++) {
        for (int i = 0; i < Cards_Delt.size(); i++) {
            hand_ hand_i = Cards_Delt[i];
            std::vector<float> cards_ = hand_i.cards();
            float chance = hand_i.p();
            int player_traversing = (i + j)% n_players;
            Decition_Node node_root = Decition_Node(last_agressor, round_raised, cards_, n_players, history, round_history, round, player_contributions, pot, player_status, player_turn, gamstate_vec, round_actions_count, max_actions_per_round, player_traversing);
            node_root.nerual_network_recursion(regrets, strategy, reach, player_traversing, rng, gamstate_training, regret_training, chance_training, chance, max_pot, max_player_contribution, ranks, WN, hidden_layer_size, n_hidden_layers, stratagy_training, max_capacity, N_samples_colected);
        }
        if (j % n_per_network_update == 0) {
            for (int j = 0; j < gernerations; j++) {
                network_count++;
                float n = n0 / (1 + k * network_count);
                for (int i = 0; i < gamstate_training.size(); i++) {
                    std::vector<float> input = gamstate_normalisation(max_pot, max_player_contribution, ranks, n_players, gamstate_training[i]);
                    input.push_back(1);
                    std::vector<float> output = regret_normalisation(max_pot, regret_training[i]);
                    float scale = chance_training[i];
                    weight_update(WN, input, hidden_layer_size, n_hidden_layers, output, scale, loss_tracking_regret, n);
                    std::vector<float> output_stratagy = stratagy_training[i];
                    weight_update(WN_stratagy, input, hidden_layer_size, n_hidden_layers, output_stratagy, scale, loss_tracking_stratagy, n);
                }
            }
            {
                float total_loss = 0.0f;
                for (int i = 0; i < (int)gamstate_training.size(); i++) {
                    std::vector<float> eval_input = gamstate_normalisation(max_pot, max_player_contribution, ranks, n_players, gamstate_training[i]);
                    eval_input.push_back(1.0f);
                    std::vector<float> eval_output = regret_normalisation(max_pot, regret_training[i]);
                    float eval_scale = chance_training[i];

                    std::vector<float> eval_l1 = eval_input * WN[0];
                    std::vector<float> eval_a1(hidden_layer_size + 1);
                    std::transform(eval_l1.begin(), eval_l1.end(), eval_a1.begin(), sigmoid);
                    eval_a1[hidden_layer_size] = 1;
                    std::vector<std::vector<float>> eval_an;
                    eval_an.push_back(eval_a1);
                    for (int k = 0; k < n_hidden_layers - 1; k++) {
                        std::vector<float> eval_lk = eval_an[k] * WN[k + 1];
                        std::vector<float> eval_ak(hidden_layer_size + 1);
                        std::transform(eval_lk.begin(), eval_lk.end(), eval_ak.begin(), sigmoid);
                        eval_ak[hidden_layer_size] = 1;
                        eval_an.push_back(eval_ak);
                    }
                    std::vector<float> eval_afinal = eval_an.back() * WN[n_hidden_layers];

                    float eval_a0 = 0;
                    for (int k = 0; k < 5; k++) eval_a0 += (eval_output[k] - eval_afinal[k]) * (eval_output[k] - eval_afinal[k]);
                    total_loss += eval_a0 * eval_scale;
                }
                float avg_loss = total_loss / (float)gamstate_training.size();
                convergence_tracking.push_back(avg_loss);
                std::cout << "[outer iter " << j << "] buffer size: " << gamstate_training.size()
                    << "  avg loss over full buffer: " << avg_loss << "\n";
            }
            {
                float total_loss = 0.0f;
                for (int i = 0; i < (int)gamstate_training.size(); i++) {
                    std::vector<float> eval_input = gamstate_normalisation(max_pot, max_player_contribution, ranks, n_players, gamstate_training[i]);
                    eval_input.push_back(1.0f);
                    std::vector<float> eval_output = stratagy_training[i];
                    float eval_scale = chance_training[i];

                    std::vector<float> eval_l1 = eval_input * WN_stratagy[0];
                    std::vector<float> eval_a1(hidden_layer_size + 1);
                    std::transform(eval_l1.begin(), eval_l1.end(), eval_a1.begin(), sigmoid);
                    eval_a1[hidden_layer_size] = 1;
                    std::vector<std::vector<float>> eval_an;
                    eval_an.push_back(eval_a1);
                    for (int k = 0; k < n_hidden_layers - 1; k++) {
                        std::vector<float> eval_lk = eval_an[k] * WN_stratagy[k + 1];
                        std::vector<float> eval_ak(hidden_layer_size + 1);
                        std::transform(eval_lk.begin(), eval_lk.end(), eval_ak.begin(), sigmoid);
                        eval_ak[hidden_layer_size] = 1;
                        eval_an.push_back(eval_ak);
                    }
                    std::vector<float> eval_afinal = eval_an.back() * WN_stratagy[n_hidden_layers];

                    float eval_a0 = 0;
                    for (int k = 0; k < 5; k++) eval_a0 += (eval_output[k] - eval_afinal[k]) * (eval_output[k] - eval_afinal[k]);
                    total_loss += eval_a0 * eval_scale;
                }
                float avg_loss = total_loss / (float)gamstate_training.size();
                convergence_tracking_stratagy.push_back(avg_loss);
                std::cout << "[outer iter " << j << "] buffer size: " << gamstate_training.size()
                    << "  avg loss over full buffer (stratagy): " << avg_loss << "\n";
            }
        }
    }

    //final normalisation
    for (auto& entry : strategy) {
        const std::string& key = entry.first;   //id   
        std::vector<float>& vec = entry.second;     //stratagy
        float vec_total = 0;
        int I = vec.size();
        for (int i = 0; i < I; i++) {
            vec_total += vec[i];
        }
        if (vec_total > 0) {
            for (int i = 0; i < I; i++) {
                vec[i] /= vec_total;
            }
        }
        else {
            for (int i = 0; i < I; i++) vec[i] = 1.0f / I;
        }
    }

    // ---- build the tabular-vs-network strategy comparison table ----
    std::unordered_map<std::string, std::vector<std::vector<float>>> network_stratagy_map;
    for (int i = 0; i < (int)Cards_Delt.size(); i++) {
        std::vector<float> cards_ = Cards_Delt[i].cards();
        float chance = Cards_Delt[i].p();
        int player_traversing = i % n_players;
        Decition_Node node_root(last_agressor, round_raised, cards_, n_players, history, round_history, round, player_contributions, pot, player_status, player_turn, gamstate_vec, round_actions_count, max_actions_per_round, player_traversing);
        node_root.final_stratagy_map_recursion(regrets, strategy, reach, player_traversing, rng, gamstate_training, regret_training, chance_training, chance, max_pot, max_player_contribution, ranks, WN_stratagy, hidden_layer_size, n_hidden_layers, stratagy_training, network_stratagy_map);
    }

    // ---- print block: worst mismatches between tabular and network strategy ----
    {
        std::vector<std::string> action_names = { "fold", "check", "call", "bet", "raise" };
        std::vector<std::pair<float, std::string>> discrepancies;
        float total_discrepancy = 0.0f;

        for (auto& kv : network_stratagy_map) {
            const std::vector<float>& cfr = kv.second[0];
            const std::vector<float>& net = kv.second[1];
            float max_diff = 0.0f;
            for (int a = 0; a < 5; a++) {
                max_diff = std::max(max_diff, std::fabs(cfr[a] - net[a]));
            }
            discrepancies.push_back({ max_diff, kv.first });
            total_discrepancy += max_diff;
        }

        std::sort(discrepancies.begin(), discrepancies.end(), [](auto& a, auto& b) { return a.first > b.first; });

        std::cout << "\n=== strategy network vs tabular CFR: " << discrepancies.size() << " infosets compared ===\n";
        std::cout << "average max-action discrepancy: " << (discrepancies.empty() ? 0.0f : total_discrepancy / discrepancies.size()) << "\n\n";

        int shown = 0;
        for (auto& d : discrepancies) {
            if (shown >= 15) break;
            const std::vector<float>& cfr = network_stratagy_map[d.second][0];
            const std::vector<float>& net = network_stratagy_map[d.second][1];
            std::cout << "key=" << d.second << "  (max diff=" << d.first << ")\n";
            std::cout << "  tabular: ";
            for (int a = 0; a < 5; a++) std::cout << action_names[a] << "=" << cfr[a] << " ";
            std::cout << "\n  network: ";
            for (int a = 0; a < 5; a++) std::cout << action_names[a] << "=" << net[a] << " ";
            std::cout << "\n\n";
            shown++;
        }
    }



    

    {
        size_t n_loss = loss_tracking_regret.size();
        size_t chunk = n_loss / 10;
        float first_avg = 0, last_avg = 0;
        for (size_t idx = 0; idx < chunk; idx++) first_avg += loss_tracking_regret[idx];
        for (size_t idx = n_loss - chunk; idx < n_loss; idx++) last_avg += loss_tracking_regret[idx];
        std::cout << "regret first 10% avg loss: " << first_avg / chunk << "\n";
        std::cout << "regret last 10% avg loss: " << last_avg / chunk << "\n";
    }
    {
        size_t n_loss = loss_tracking_stratagy.size();
        size_t chunk = n_loss / 10;
        float first_avg = 0, last_avg = 0;
        for (size_t idx = 0; idx < chunk; idx++) first_avg += loss_tracking_stratagy[idx];
        for (size_t idx = n_loss - chunk; idx < n_loss; idx++) last_avg += loss_tracking_stratagy[idx];
        std::cout << "strategy first 10% avg loss: " << first_avg / chunk << "\n";
        std::cout << "strategy last 10% avg loss: " << last_avg / chunk << "\n";
    }

    //#####################################################################################################
    //#####################################################################################################
    //#####################################################################################################



    return 0;
}