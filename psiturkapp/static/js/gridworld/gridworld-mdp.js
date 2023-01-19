import {product, choose, fromPairs} from "./utils.js"

export class GridWorldMDP {
    constructor ({
        tile_array,
        feature_rewards = {},
        absorbing_features=["g",],
        wall_features=["#",],
        default_features=[".",],
        initial_features=["s",],
        step_cost=0,
        wall_bump_cost=0,
        success_prob=1.0,
        discount_rate=1.0,
        include_wait=false
    }) {
        this.height = tile_array.length;
        this.width = tile_array[0].length;
        this.walls = [];
        this.absorbing_locations = [];
        this.initial_locations = [];
        this.locations = [];
        this.location_features = {};
        this.feature_locations = {};
        for (let x = 0; x < this.width; x++) {
            for (let y = 0; y < this.height; y++) {
                let loc = String([x, y]);
                let f = tile_array[this.height - y - 1][x];
                this.locations.push(loc)
                this.location_features[loc] = f
                if (!(f in this.feature_locations)) {
                    this.feature_locations[f] = [];
                }
                this.feature_locations[f].push(loc)
                if (wall_features.includes(f)) {
                    this.walls.push(loc)
                }
                if (absorbing_features.includes(f)) {
                    this.absorbing_locations.push(loc)
                }
                if (initial_features.includes(f)) {
                    this.initial_locations.push(loc)
                }
            }
        }
        this.actions = [
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1],
        ]
        if (include_wait) {
            this.actions.push([0, 0])
        }

        this.initial_features = initial_features;
        this.feature_rewards = feature_rewards;
        this.step_cost = step_cost;
        this.wall_bump_cost = wall_bump_cost;
        this.wall_features = wall_features;
    }

    // location properties
    is_wall(loc) {
        loc = String(loc)
        return this.wall_features.includes(this.location_features[loc])
    }

    is_on_grid(loc) {
        if (typeof(loc) === 'string') {
            loc = loc.split(",").map(Number);
        }
        return (0 <= loc[0]) && (loc[0] < this.width) && (0 <= loc[1]) && (loc[1] < this.height)
    }

    is_absorbing(loc) {
        loc = String(loc);
        return this.absorbing_locations.includes(loc);
    }

    // applying actions and transitions
    apply_action(state, action) {
        if (typeof(state) === 'string') {
            state = state.split(",").map(Number);
        }
        if (typeof(action) === 'string') {
            action = action.split(",").map(Number);
        }
        return [state[0] + action[0], state[1] + action[1]]
    }

    transition_reward({state, action}) {
        let r = this.step_cost;
        let ns = this.apply_action(state, action);
        if (this.is_wall(ns) || !this.is_on_grid(ns)) {
            ns = state;
            r += this.wall_bump_cost;
        }
        let ns_f = this.location_features[ns];
        if (Object.keys(this.feature_rewards).includes(ns_f)) {
            r += this.feature_rewards[ns_f]
        }
        return [ns, r]
    }

    transition ({state, action}) {
        // We assume deterministic transitions/rewards
        let [ns, r] = this.transition_reward({state, action});
        return ns
    }

    reward ({state, action, nextstate}) {
        // We assume deterministic transitions/rewards
        let [ns, r] = this.transition_reward({state, action});
        return r
    }

    initial_state () {
        // only return first initial state
        if (this.initial_locations.length === 0) {
            return undefined
        }
        return this.initial_locations[0].split(",").map(Number)
    }
}
