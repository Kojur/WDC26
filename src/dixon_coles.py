"""Dixon-Coles bivariate-Poisson model for football scoreline prediction."""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


def dc_tau(x, y, lam, mu, rho):
    """Dixon-Coles low-score correction (scalar)."""
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _vec_tau(x, y, lam, mu, rho):
    """Vectorized Dixon-Coles correction over arrays of scores/rates."""
    tau = np.ones_like(lam, dtype=float)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho
    return tau


def _neg_log_likelihood(params, hi, ai, x, y, neutral, weights, n_teams):
    attack = params[:n_teams]
    defense = params[n_teams:2 * n_teams]
    gamma = params[2 * n_teams]
    rho = params[2 * n_teams + 1]
    home_term = gamma * (~neutral)
    lam = np.exp(attack[hi] + defense[ai] + home_term)
    mu = np.exp(attack[ai] + defense[hi])
    tau = np.clip(_vec_tau(x, y, lam, mu, rho), 1e-10, None)
    ll = weights * (np.log(tau) + poisson.logpmf(x, lam) + poisson.logpmf(y, mu))
    return -float(np.sum(ll))


class DixonColesModel:
    """Maximum-likelihood Dixon-Coles model with time-decay weighting.

    Fitted attributes:
        attack:  per-team attacking strength (higher => scores more goals).
        defense: per-team defensive parameter beta (DC 1997). A *more negative*
                 value means the team concedes fewer goals (stronger defence).
                 So overall team strength is ``attack - defense``.
        home_adv: home-advantage term gamma (added to lambda for non-neutral games).
        rho: low-score correction parameter.
    """

    def __init__(self, max_goals=10):
        self.max_goals = max_goals
        self.teams = None
        self.team_index = None
        self.attack = None
        self.defense = None
        self.home_adv = None
        self.rho = None
        self._mean_defense = None

    def fit(self, matches, xi=0.0, ref_date=None):
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi = matches["home_team"].map(idx).to_numpy()
        ai = matches["away_team"].map(idx).to_numpy()
        x = matches["home_score"].to_numpy().astype(int)
        y = matches["away_score"].to_numpy().astype(int)
        neutral = matches["neutral"].to_numpy().astype(bool)
        ref_date = matches["date"].max() if ref_date is None else ref_date
        age = (ref_date - matches["date"]).dt.days.to_numpy() / 365.25
        weights = np.exp(-xi * age)

        p0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.05]])
        res = minimize(
            _neg_log_likelihood, p0,
            args=(hi, ai, x, y, neutral, weights, n),
            method="L-BFGS-B",
        )
        att = res.x[:n]
        dfn = res.x[n:2 * n]
        c = att.mean()                 # recenter for identifiability
        att, dfn = att - c, dfn + c
        self.teams = teams
        self.team_index = idx
        self.attack = att
        self.defense = dfn
        self.home_adv = float(res.x[2 * n])
        self.rho = float(res.x[2 * n + 1])
        self._mean_defense = float(dfn.mean())
        self.result_ = res
        return self

    def _params(self, team):
        if team in self.team_index:
            i = self.team_index[team]
            return float(self.attack[i]), float(self.defense[i])
        return 0.0, self._mean_defense  # global-average cold-start fallback

    def expected_goals(self, home, away, neutral=True):
        a_h, d_h = self._params(home)
        a_a, d_a = self._params(away)
        gamma = 0.0 if neutral else self.home_adv
        lam = float(np.exp(a_h + d_a + gamma))
        mu = float(np.exp(a_a + d_h))
        return lam, mu

    def score_matrix(self, home, away, neutral=True):
        lam, mu = self.expected_goals(home, away, neutral)
        g = np.arange(self.max_goals + 1)
        mat = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        mat[0, 0] *= 1.0 - lam * mu * self.rho
        mat[0, 1] *= 1.0 + lam * self.rho
        mat[1, 0] *= 1.0 + mu * self.rho
        mat[1, 1] *= 1.0 - self.rho
        mat = np.clip(mat, 0.0, None)  # guard against negative tau at extreme rho
        return mat / mat.sum()

    def predict_result(self, home, away, neutral=True):
        mat = self.score_matrix(home, away, neutral)
        return {
            "home_win": float(np.tril(mat, -1).sum()),  # home goals > away goals
            "draw": float(np.trace(mat)),
            "away_win": float(np.triu(mat, 1).sum()),
        }

    def sample_scoreline(self, home, away, rng, neutral=True):
        mat = self.score_matrix(home, away, neutral)
        flat = mat.ravel()
        k = rng.choice(flat.size, p=flat)
        hg, ag = divmod(int(k), mat.shape[1])
        return hg, ag
