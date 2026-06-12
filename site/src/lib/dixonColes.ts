import type { Ratings } from "../data/types";

export function poissonPmf(k: number, lambda: number): number {
  let fact = 1;
  for (let i = 2; i <= k; i++) fact *= i;
  return (Math.exp(-lambda) * Math.pow(lambda, k)) / fact;
}

function params(team: string, r: Ratings): [number, number] {
  if (Object.prototype.hasOwnProperty.call(r.attack, team)) {
    return [r.attack[team], r.defense[team]];
  }
  return [0, r.mean_defense];
}

export function expectedGoals(
  home: string, away: string, r: Ratings, neutral = true,
): [number, number] {
  const [aH, dH] = params(home, r);
  const [aA, dA] = params(away, r);
  const gamma = neutral ? 0 : r.home_adv;
  return [Math.exp(aH + dA + gamma), Math.exp(aA + dH)];
}

export function scoreMatrix(
  home: string, away: string, r: Ratings, neutral = true,
): number[][] {
  const [lam, mu] = expectedGoals(home, away, r, neutral);
  const n = r.max_goals + 1;
  const mat: number[][] = [];
  for (let x = 0; x < n; x++) {
    mat[x] = [];
    for (let y = 0; y < n; y++) mat[x][y] = poissonPmf(x, lam) * poissonPmf(y, mu);
  }
  mat[0][0] *= 1 - lam * mu * r.rho;
  mat[0][1] *= 1 + lam * r.rho;
  mat[1][0] *= 1 + mu * r.rho;
  mat[1][1] *= 1 - r.rho;
  let sum = 0;
  for (let x = 0; x < n; x++)
    for (let y = 0; y < n; y++) { if (mat[x][y] < 0) mat[x][y] = 0; sum += mat[x][y]; }
  for (let x = 0; x < n; x++)
    for (let y = 0; y < n; y++) mat[x][y] /= sum;
  return mat;
}

export interface Result { homeWin: number; draw: number; awayWin: number; }

export function predictResult(mat: number[][]): Result {
  let homeWin = 0, draw = 0, awayWin = 0;
  for (let x = 0; x < mat.length; x++)
    for (let y = 0; y < mat[x].length; y++) {
      if (x > y) homeWin += mat[x][y];
      else if (x === y) draw += mat[x][y];
      else awayWin += mat[x][y];
    }
  return { homeWin, draw, awayWin };
}

export function mostLikelyScore(mat: number[][]): { home: number; away: number; prob: number } {
  let best = -1, home = 0, away = 0;
  for (let x = 0; x < mat.length; x++)
    for (let y = 0; y < mat[x].length; y++)
      if (mat[x][y] > best) { best = mat[x][y]; home = x; away = y; }
  return { home, away, prob: best };
}
