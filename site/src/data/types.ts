export interface Ratings {
  teams: string[];
  attack: Record<string, number>;
  defense: Record<string, number>;
  home_adv: number;
  rho: number;
  max_goals: number;
  mean_defense: number;
  fit: { min_date: string; xi: number; n_matches: number };
}

export interface TeamOdds {
  team: string;
  p_r32: number; p_r16: number; p_qf: number;
  p_sf: number; p_final: number; p_champion: number;
}

export interface Simulation { n_sims: number; seed: number; teams: TeamOdds[]; }

export type Groups = Record<string, string[]>;

export interface Meta {
  dataset: { n_matches: number; n_teams: number; date_min: string; date_max: string };
  ratings_sanity: { fifa_rank_corr: number; n_matched: number };
  backtest: {
    wc_years: number[]; n_matches: number;
    model: { rps: number; log_loss: number; accuracy: number };
    baseline: { rps: number; log_loss: number; accuracy: number };
    best_xi: number;
  };
  model: { home_adv: number; rho: number };
}

export interface ParityCase {
  home: string; away: string; neutral: boolean;
  home_win: number; draw: number; away_win: number;
}
