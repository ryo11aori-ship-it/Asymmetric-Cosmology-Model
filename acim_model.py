import numpy as np
from scipy.stats import dirichlet, norm, entropy
from scipy.special import rel_
entr
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Set
class AsymmetricConsciousnessModel:
def
init
__
__(self, num_nodes: int = 10, recursion_depth: int = 5,
base
_dimension: float = 3.0, meta_dimension: float = 5.0,
real
_data: bool = False, fMRI_data: Dict = None, projection_factor: float = 0.7):
self.num
nodes = num
nodes
_
_
self.recursion
_depth = recursion_depth
self.base
dimension = base
dimension
_
_
self.meta
dimension = meta
dimension
_
_
self.real
data = real
data
_
_
self.fMRI
data = fMRI
_
_data or {}
self.graph = self._
build
_graph()
self.prior_
distributions = self.
_
self.observed
_distributions = {}
initialize
_distributions()
self.delta
_obs = {}
self.O
_values = {}
self.AII = 0.0
self.effective
_dimensions = {i: base_dimension for i in range(num_nodes)}
self.aii
_history = []
self.informational
_gravity_
contribution = 0.0
self.state
_params = {'noise_level': 0.0, 'edge_reduction': 1.0}
self.dimension
_groups = self._
initialize
dimension
_
_groups()
self.complexity_costs = {}
self.projection_factor = projection_
factor # プロパティとして保持
def
build
_
_graph(self) -> nx.DiGraph:
G = nx.DiGraph()
for i in range(self.num_nodes):
dim = self.fMRI
_data.get(f"node_{i}", {}).get("dimension",
np.random.uniform(self.base_dimension - 1, self.meta_dimension + 1))
G.add
_node(i, dimension=dim)
if self.real
data and self.fMRI
data:
_
_
for edge in self.fMRI_data.get("edges", []):
i, j, w = edge["source"], edge["target"], edge.get("weight", np.random.uniform(0.5,
1.0))
if G.nodes[i]["dimension"] > G.nodes[j]["dimension"]:
G.add
_edge(i, j, weight=w * self.state_params['edge_
reduction'])
else:
for i in range(self.num_nodes):
for j in range(self.num_nodes):
if i != j and G.nodes[i]["dimension"] > G.nodes[j]["dimension"]:
if np.random.rand() > 0.5:
G.add
_edge(i, j, weight=np.random.uniform(0.5, 1.0) *
self.state
_params['edge_
reduction'])
return G
def
initialize
dimension
_
_
_groups(self) -> Dict[float, Set[int]]:
groups = {self.base_dimension: set(), self.meta_dimension: set()}
for node in self.graph.nodes:
dim = self.graph.nodes[node]["dimension"]
if dim <= (self.base_
dimension + self.meta
_dimension) / 2:
groups[self.base
_
dimension].add(node)
else:
groups[self.meta
_
dimension].add(node)
return groups
def
initialize
_
_distributions(self) -> Dict[int, np.ndarray]:
distributions = {}
for node in self.graph.nodes:
alpha = np.ones(5) * 10
distributions[node] = dirichlet.rvs(alpha, size=1)[0] + self.state
_params['noise
_
level'] *
np.random.randn(5)
distributions[node] /= distributions[node].sum()
return distributions
def
functor
_
_observation(self, observer: int, observed: int, prior: np.ndarray) ->
Tuple[np.ndarray, float, float]:
"""圏論的観測射 (Functor): 次元間の非対称的変換とコストをモデル化"""
delta
_dim = self.graph.nodes[observer]["dimension"] - self.graph.nodes[observed]
["dimension"]
intervention = self.graph[observer][observed].get("weight", 0.5) * delta_
dim
post = prior + norm×rvs(0, intervention, size=prior.shape) + self.state_params['noise
_
level']
* np.random.randn(5)
post = np×clip(post, 1e-10, np.inf)
post /= post.sum()
# 次元間の関手変換: 高次元への射影とコスト計算
complexity_
cost = 0.0
if self.graph.nodes[observer]["dimension"] > self.graph.nodes[observed]["dimension"]:
post = post × self×projection_factor + (1 - self.projection_factor) * np.mean(post)
complexity_cost = np×sum(np×abs(prior - post)) / len(prior)
elif self.graph.nodes[observer]["dimension"] < self.graph.nodes[observed]["dimension"]:
loss
factor = 0.3
_
complexity_
cost = loss
_factor * np.sum(prior) / len(prior)
kl
_div = np.sum(rel_entr(post, prior))
O = kl
_div / (1 + kl_div) if kl_
div > 0 else 0.0
return post, O, complexity_
cost
def
_perform_observation(self, level: int = 0):
if level >= self.recursion
_depth:
return
current
_observed = {}
current
_delta = {}
current
_O = {}
current
_complexity = {}
for edge in self×graph.edges(data=True):
observer, observed, data = edge
prior = self.prior_
distributions[observed]
post, O, complexity = self._
functor
_observation(observer, observed, prior)
delta = np.sum(rel_entr(post, prior))
current
_
observed[(observer, observed)] = post
current
_
delta[(observer, observed)] = delta
current
_
O[(observer, observed)] = O
current
_complexity[(observer, observed)] = complexity
alpha = 0.1
self.effective
_
dimensions[observed] ×= (1 + alpha * O * abs(intervention))
self.graph[observer][observed]['weight'] *= (1 + alpha * O)
self.observed
_
distributions[level] = current
observed
_
self.delta
_
obs[level] = current
delta
_
self.O
_
values[level] = current
O
_
self.complexity_
costs[level] = current
_complexity
next
_prior_distributions = self.prior_distributions.copy()
for (observer, observed), post_
dist in current
_observed.items():
next
_prior_
distributions[observed] = post_
dist
self.prior_
distributions = next
_prior_
distributions
self.
_perform_observation(level + 1)
def compute_AII(self) -> float:
if not self.delta
obs:
_
self.
_perform_observation()
integrated_
delta = 0.0
total
_complexity = 0.0
for level in self.delta
obs:
_
level
_deltas = list(self.delta_
obs[level].values())
level
_complexities = list(self.complexity_
costs[level].values())
if level
deltas and level
_
_complexities:
integrated_delta += np.exp(np.mean(level_deltas))
total
_complexity += np.mean(level_complexities) * len(level_deltas)
self.AII = integrated_delta / self.recursion_depth + total_complexity / self.num_
nodes
self.aii
_history.append(self.AII)
return self.AII
def simulate
_evolution(self, iterations: int = 100, state: str = 'normal'):
if state == 'anesthesia':
self.state
_params = {'noise_level': 0.5, 'edge_reduction': 0.7}
elif state == 'sleep':
self.state
_params = {'noise_level': 0.3, 'edge_reduction': 0.9}
else:
self.state
_params = {'noise_level': 0.0, 'edge_reduction': 1.0}
for
_ in range(iterations):
self.
_perform_observation()
self.informational
_gravity_contribution += self.AII * np.exp(-self.num_nodes)
def load
_
self.fMRI
fMRI
_data(self, data: Dict):
data = data
_
self.real
data = True
_
self.graph = self._
build
_graph()
self.dimension
_groups = self._
initialize
dimension
_
_groups()
def correlate
with
CRS
_
_
_R(self, crs_
r
_
scores: List[float]):
if not self.aii
_history:
self.simulate
_evolution()
if len(self.aii_history) < len(crs_
r
_scores):
raise ValueError("AII history length must match CRS-R scores length for correlation.")
from scipy.stats import pearsonr
corr, _ = pearsonr(self.aii_history[:len(crs_
r
_scores)], crs_
r
_scores)
return corr
def optimize_projection_factor(self, crs_
r
_
scores: List[float], step: float = 0.1):
"""射影因子の最適化: CRS-Rスコアとの相関を最大化"""
best
factor = 0.1
_
best
corr = -1.0
_
for factor in np.arange(0.1, 1.0 + step, step):
self.projection_
factor = factor # ループ内でモデルのファクターを更新
self.aii
_history = []
self.simulate
_evolution(iterations=50, state='normal')
corr = self.correlate
with
CRS
_
_
_R(crs_
r
_scores)
if corr > best
corr:
_
best
corr = corr
_
best
factor = factor
_
return best
print(f"最適な projection_factor: {best_factor}, 最大相関: {best_corr}")
factor
_
def visualize(self):
pos = nx.spring_layout(self.graph)
nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', arrows=True)
plt.title("Asymmetric Observation Network with Functor Mapping")
plt.show()
plt.plot(range(len(self.aii_history)), self.aii_history)
plt.title("AII Evolution Over Time")
plt.xlabel("Time Step")
plt.ylabel("AII Value")
plt.show()
# 使用例
if
name
== "
main
":
__
__
__
__
model = AsymmetricConsciousnessModel(num_nodes=20, recursion_depth=10,
base
_dimension=3.0, meta_dimension=5.0)
model.simulate
_evolution(iterations=50, state='normal')
model.visualize()
print(f"Final AII: {model.AII}")
print(f"Informational Gravity Contribution: {model.informational_gravity_contribution}")
# 麻酔状態のシミュレーション
model
_anesthesia = AsymmetricConsciousnessModel(num_nodes=20, recursion_depth=10)
model
anesthesia.simulate
_
_evolution(iterations=50, state='anesthesia')
model
_anesthesia.visualize()
print(f"AII under Anesthesia: {model_anesthesia.AII}")
# fMRIデータ例
fMRI
_data = {
"nodes": [{"id": i, "dimension": np.random.uniform(3, 5)} for i in range(10)],
"edges": [{"source": i, "target": j, "weight": np.random.uniform(0.5, 1.0)}
for i in range(10) for j in range(10) if i != j and np.random.rand() > 0.7]
}
model
_
model
_
with
_data = AsymmetricConsciousnessModel(real_data=True, fMRI_
data=fMRI
_data)
with
data.simulate
_
_evolution()
model
with
_
_data.visualize()
# CRS-R相関検証と射影因子の最適化
crs
_
r = [5, 6, 7, 8, 9] # 仮想的データ
optimal_
factor = model
with
_
_data.optimize_projection_factor(crs_r)
print(f"最適化された projection_factor: {optimal_factor}")
model
with
data.correlate
with
CRS
_
_
_
_
_R(crs_r)
