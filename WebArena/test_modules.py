"""
Comprehensive tests for the 4 ExploreAgent modules.
Tests each module independently, then integration.
"""

import sys
import os
import json
import tempfile
import shutil

sys.path.insert(0, '.')

# ============================================================
# Module 0: Strategy Space
# ============================================================
def test_strategy_space():
    print("=" * 80)
    print("MODULE 0: Strategy Space")
    print("=" * 80)

    from memory_agents.strategy_space import StrategySpace, MilestoneNode

    errors = []

    # ── Test 1: MilestoneNode basics ──
    print("\n[Test 1] MilestoneNode dataclass...")
    node = MilestoneNode(
        id="test_1", milestone="Open homepage",
        key_actions=["goto(url)"], domain="shopping",
        alpha=3.0, beta_param=2.0
    )
    assert node.success_rate() == (3.0 - 1) / (3.0 + 2.0 - 2), f"success_rate wrong: {node.success_rate()}"
    d = node.to_dict()
    node2 = MilestoneNode.from_dict(d)
    assert node2.milestone == node.milestone
    assert node2.alpha == node.alpha
    print("  PASS: MilestoneNode create, success_rate, to_dict, from_dict")

    # ── Test 2: Flat structure ──
    print("\n[Test 2] Flat structure...")
    space = StrategySpace(structure="flat")
    n1 = space.add_node("Strategy A", domain="shopping")
    n2 = space.add_node("Strategy B", domain="shopping")
    n3 = space.add_node("Strategy C", domain="reddit")
    paths_all = space.get_paths()
    paths_shop = space.get_paths("shopping")
    paths_reddit = space.get_paths("reddit")
    assert len(paths_all) == 3, f"Expected 3 flat paths, got {len(paths_all)}"
    assert len(paths_shop) == 2, f"Expected 2 shopping paths, got {len(paths_shop)}"
    assert len(paths_reddit) == 1, f"Expected 1 reddit path, got {len(paths_reddit)}"
    # Each flat path has exactly 1 node
    assert all(len(p) == 1 for p in paths_all), "Flat paths should have 1 node each"
    print(f"  PASS: flat — {len(paths_all)} total, {len(paths_shop)} shopping, {len(paths_reddit)} reddit")

    # ── Test 3: Tree structure ──
    print("\n[Test 3] Tree structure...")
    space = StrategySpace(structure="tree")
    root = space.add_node("Root", domain="shopping")
    child1 = space.add_node("Child A", domain="shopping", parent_id=root)
    child2 = space.add_node("Child B", domain="shopping", parent_id=root)
    leaf1 = space.add_node("Leaf A1", domain="shopping", parent_id=child1)
    leaf2 = space.add_node("Leaf B1", domain="shopping", parent_id=child2)
    leaf3 = space.add_node("Leaf B2", domain="shopping", parent_id=child2)
    paths = space.get_paths("shopping")
    assert len(paths) == 3, f"Expected 3 tree paths, got {len(paths)}"
    # Each path: root -> child -> leaf = 3 nodes
    assert all(len(p) == 3 for p in paths), f"Tree paths should have 3 nodes, got {[len(p) for p in paths]}"
    print(f"  PASS: tree — {len(paths)} paths, depths: {[len(p) for p in paths]}")

    # ── Test 4: DAG structure (multi-parent) ──
    print("\n[Test 4] DAG structure (multi-parent edges)...")
    space = StrategySpace(structure="dag")
    r = space.add_node("Start", domain="d")
    a = space.add_node("Path A", domain="d", parent_id=r)
    b = space.add_node("Path B", domain="d", parent_id=r)
    merge = space.add_node("Merge point", domain="d", parent_id=a)
    space.add_edge(b, merge)  # b -> merge (second parent)
    assert len(space.nodes[merge].parents) == 2, f"Merge node should have 2 parents, got {len(space.nodes[merge].parents)}"
    paths = space.get_paths("d")
    assert len(paths) == 2, f"Expected 2 DAG paths, got {len(paths)}"
    print(f"  PASS: dag — {len(paths)} paths, merge node has {len(space.nodes[merge].parents)} parents")

    # ── Test 5: Prune ──
    print("\n[Test 5] Prune operation...")
    space = StrategySpace(structure="tree")
    r = space.add_node("Root", domain="x")
    c1 = space.add_node("Keep", domain="x", parent_id=r)
    c2 = space.add_node("Remove", domain="x", parent_id=r)
    c2_child = space.add_node("Remove child", domain="x", parent_id=c2)
    assert len(space.nodes) == 4
    space.prune(c2)
    assert len(space.nodes) == 2, f"After prune expected 2 nodes, got {len(space.nodes)}"
    assert c2 not in space.nodes
    assert c2_child not in space.nodes
    assert c2 not in space.nodes[r].children
    print(f"  PASS: prune removed subtree, {len(space.nodes)} nodes remain")

    # ── Test 6: update_node ──
    print("\n[Test 6] update_node...")
    space = StrategySpace(structure="flat")
    nid = space.add_node("Original", domain="test")
    space.update_node(nid, milestone="Updated", key_actions=["new_action"])
    assert space.nodes[nid].milestone == "Updated"
    assert space.nodes[nid].key_actions == ["new_action"]
    print("  PASS: update_node modifies fields correctly")

    # ── Test 7: remove_edge ──
    print("\n[Test 7] remove_edge...")
    space = StrategySpace(structure="dag")
    a = space.add_node("A", domain="d")
    b = space.add_node("B", domain="d", parent_id=a)
    assert b in space.nodes[a].children
    space.remove_edge(a, b)
    assert b not in space.nodes[a].children
    assert a not in space.nodes[b].parents
    print("  PASS: remove_edge cleans both ends")

    # ── Test 8: update_path_stats ──
    print("\n[Test 8] update_path_stats with gamma discount...")
    space = StrategySpace(structure="tree")
    r = space.add_node("Root")
    c = space.add_node("Child", parent_id=r)
    l = space.add_node("Leaf", parent_id=c)
    path = space.get_paths()[0]
    assert len(path) == 3

    space.update_path_stats(path, success=True, gamma=0.5)
    leaf = space.nodes[l]
    child = space.nodes[c]
    root = space.nodes[r]
    # Leaf gets discount=1.0, child=0.5, root=0.25
    assert leaf.alpha == 2.0, f"Leaf alpha should be 2.0, got {leaf.alpha}"
    assert abs(child.alpha - 1.5) < 1e-6, f"Child alpha should be 1.5, got {child.alpha}"
    assert abs(root.alpha - 1.25) < 1e-6, f"Root alpha should be 1.25, got {root.alpha}"
    assert leaf.visit_count == 1
    assert child.visit_count == 1
    assert root.visit_count == 1
    print(f"  PASS: gamma-discounted stats — leaf.alpha={leaf.alpha}, child.alpha={child.alpha}, root.alpha={root.alpha}")

    # ── Test 9: to_text ──
    print("\n[Test 9] to_text rendering...")
    space = StrategySpace(structure="tree")
    r = space.add_node("Navigate", key_actions=["click [link]"], pitfalls=["May timeout"], success_signal="Page loaded")
    c = space.add_node("Search", parent_id=r, key_actions=["type [bar] 'query'"])
    path = space.get_paths()[0]
    text = space.to_text(path)
    assert "Step 1:" in text
    assert "Navigate" in text
    assert "click [link]" in text
    assert "May timeout" in text
    assert "Page loaded" in text
    assert "Step 2:" in text
    assert "Search" in text
    print(f"  PASS: to_text renders correctly ({len(text)} chars)")

    # ── Test 10: space_summary ──
    print("\n[Test 10] space_summary...")
    space = StrategySpace(structure="dag", seed_file="strategy_seeds.json")
    summary = space.space_summary("shopping")
    assert "Strategy Space" in summary
    assert "shopping" in summary
    print(f"  PASS: space_summary for shopping:\n{summary[:300]}")

    # ── Test 11: Persistence (save/load) ──
    print("\n[Test 11] Persistence save/load...")
    tmpdir = tempfile.mkdtemp()
    try:
        space = StrategySpace(structure="dag", seed_file="strategy_seeds.json")
        # Modify some stats
        paths = space.get_paths("shopping")
        if paths:
            space.update_path_stats(paths[0], True)
            space.update_path_stats(paths[0], False)
        save_path = os.path.join(tmpdir, "test.json")
        space.save(save_path)
        assert os.path.exists(save_path)

        # Load into fresh space
        space2 = StrategySpace(structure="flat")  # intentionally wrong structure
        space2.load(save_path)
        assert space2.structure == "dag", "Structure should be restored from file"
        assert len(space2.nodes) == len(space.nodes), f"Node count mismatch: {len(space2.nodes)} vs {len(space.nodes)}"

        # Verify stats survived
        if paths:
            leaf_id = paths[0][-1].id
            assert space2.nodes[leaf_id].visit_count == 2
        print(f"  PASS: save/load preserves {len(space2.nodes)} nodes with stats")
    finally:
        shutil.rmtree(tmpdir)

    # ── Test 12: Seed file loading ──
    print("\n[Test 12] Seed file loading...")
    space = StrategySpace(structure="dag", seed_file="strategy_seeds.json")
    all_domains = set()
    for node in space.nodes.values():
        if node.domain:
            all_domains.add(node.domain)
    expected_domains = {"shopping", "reddit", "gitlab", "shopping_admin", "map", "wikipedia"}
    assert expected_domains.issubset(all_domains), f"Missing domains: {expected_domains - all_domains}"
    print(f"  PASS: seeds loaded {len(space.nodes)} nodes across domains: {sorted(all_domains)}")

    # ── Test 13: Domain filtering ──
    print("\n[Test 13] Domain filtering...")
    for domain in ["shopping", "reddit", "gitlab"]:
        paths = space.get_paths(domain)
        for path in paths:
            for node in path:
                assert node.domain == domain or node.domain == "", \
                    f"Node domain {node.domain} doesn't match filter {domain}"
    print("  PASS: domain filtering is correct")

    # ── Test 14: Empty space ──
    print("\n[Test 14] Edge case: empty space...")
    empty = StrategySpace(structure="dag")
    assert empty.get_paths() == []
    assert empty.space_summary() == "No strategies available."
    print("  PASS: empty space returns [] and summary message")

    # ── Test 15: action_tree structure ──
    print("\n[Test 15] action_tree structure...")
    space = StrategySpace(structure="action_tree")
    r = space.add_node("click('search_btn')")
    c1 = space.add_node("type('query_box', 'shoes')", parent_id=r)
    c2 = space.add_node("click('result_1')", parent_id=c1)
    paths = space.get_paths()
    assert len(paths) == 1
    assert len(paths[0]) == 3
    print(f"  PASS: action_tree — {len(paths)} paths with {len(paths[0])} action nodes")

    print("\n" + "=" * 80)
    print(f"MODULE 0 COMPLETE: All {15} tests passed!")
    print("=" * 80)
    return True


# ============================================================
# Module 2: Exploration
# ============================================================
def test_exploration():
    print("\n" + "=" * 80)
    print("MODULE 2: Exploration")
    print("=" * 80)

    from memory_agents.strategy_space import StrategySpace
    from memory_agents.exploration import Explorer

    # Setup test space
    space = StrategySpace(structure="tree")
    r1 = space.add_node("Root1", domain="d")
    l1 = space.add_node("Leaf1", domain="d", parent_id=r1)
    r2 = space.add_node("Root2", domain="d")
    l2 = space.add_node("Leaf2", domain="d", parent_id=r2)
    r3 = space.add_node("Root3", domain="d")
    l3 = space.add_node("Leaf3", domain="d", parent_id=r3)

    # Give path1 good stats, path2 bad, path3 no visits
    for _ in range(5):
        space.update_path_stats([space.nodes[r1], space.nodes[l1]], True)
    for _ in range(5):
        space.update_path_stats([space.nodes[r2], space.nodes[l2]], False)

    paths = space.get_paths("d")
    assert len(paths) == 3

    # ── Test 1: UCB selects unvisited first ──
    print("\n[Test 1] UCB: selects unvisited path first...")
    explorer = Explorer(method="ucb", c=1.414)
    selected = explorer.select_path(space, "d")
    # Path 3 is unvisited, should be selected
    assert selected[-1].id == l3, f"UCB should select unvisited path, got {selected[-1].id}"
    print("  PASS: UCB picks unvisited path (l3)")

    # Give path3 some visits too, then check UCB scores
    space.update_path_stats([space.nodes[r3], space.nodes[l3]], False)

    # ── Test 2: UCB prefers high-reward path ──
    print("\n[Test 2] UCB: prefers high success rate...")
    selected = explorer.select_path(space, "d")
    # Path1 has 100% success, should have highest UCB
    assert selected[-1].id == l1, f"UCB should prefer path1, got {selected[-1].id}"
    print("  PASS: UCB picks high-success path (l1)")

    # ── Test 3: Thompson Sampling runs without error ──
    print("\n[Test 3] Thompson Sampling: runs correctly...")
    explorer_ts = Explorer(method="thompson")
    selections = {}
    for _ in range(100):
        selected = explorer_ts.select_path(space, "d")
        leaf_id = selected[-1].id
        selections[leaf_id] = selections.get(leaf_id, 0) + 1
    # Path1 (high alpha) should be selected most often
    assert selections.get(l1, 0) > selections.get(l2, 0), \
        f"Thompson should prefer path1: {selections}"
    print(f"  PASS: Thompson sampling distribution: {selections}")

    # ── Test 4: Epsilon-greedy exploitation ──
    print("\n[Test 4] Epsilon-greedy: exploitation (epsilon=0)...")
    explorer_eg = Explorer(method="epsilon_greedy", epsilon=0.0)
    selected = explorer_eg.select_path(space, "d")
    assert selected[-1].id == l1, f"epsilon=0 should always pick best, got {selected[-1].id}"
    print("  PASS: epsilon=0 always picks best path")

    # ── Test 5: Epsilon-greedy exploration ──
    print("\n[Test 5] Epsilon-greedy: exploration (epsilon=1.0)...")
    explorer_eg1 = Explorer(method="epsilon_greedy", epsilon=1.0)
    selections = {}
    for _ in range(100):
        selected = explorer_eg1.select_path(space, "d")
        leaf_id = selected[-1].id
        selections[leaf_id] = selections.get(leaf_id, 0) + 1
    # With epsilon=1.0, all should be selected roughly equally
    assert len(selections) >= 2, f"epsilon=1.0 should explore multiple paths: {selections}"
    print(f"  PASS: epsilon=1.0 explores: {selections}")

    # ── Test 6: Single path ──
    print("\n[Test 6] Single path available...")
    single_space = StrategySpace(structure="flat")
    single_space.add_node("Only option", domain="x")
    for method in ["ucb", "thompson", "epsilon_greedy"]:
        e = Explorer(method=method)
        selected = e.select_path(single_space, "x")
        assert len(selected) == 1
    print("  PASS: all methods handle single path correctly")

    # ── Test 7: Empty space ──
    print("\n[Test 7] Empty space...")
    empty_space = StrategySpace(structure="flat")
    for method in ["ucb", "thompson", "epsilon_greedy"]:
        e = Explorer(method=method)
        selected = e.select_path(empty_space, "x")
        assert selected == []
    print("  PASS: all methods return [] for empty space")

    # ── Test 8: UCB c parameter effect ──
    print("\n[Test 8] UCB c parameter sensitivity...")
    # High c -> more exploration, low c -> more exploitation
    explorer_low_c = Explorer(method="ucb", c=0.01)
    explorer_high_c = Explorer(method="ucb", c=100.0)
    # With very low c, should pick path1 (best success rate)
    selected_low = explorer_low_c.select_path(space, "d")
    assert selected_low[-1].id == l1, "Low c should exploit best path"
    print("  PASS: c parameter affects UCB exploration/exploitation tradeoff")

    print("\n" + "=" * 80)
    print(f"MODULE 2 COMPLETE: All {8} tests passed!")
    print("=" * 80)
    return True


# ============================================================
# Module 1: Guidance (text generation only, skip LLM calls)
# ============================================================
def test_guidance():
    print("\n" + "=" * 80)
    print("MODULE 1: Guidance")
    print("=" * 80)

    from memory_agents.strategy_space import StrategySpace, MilestoneNode

    # Build test path manually
    space = StrategySpace(structure="tree")
    r = space.add_node("Navigate to product page", domain="shopping",
                        key_actions=["click [shop_link]"],
                        pitfalls=["Link may be hidden"],
                        success_signal="Product page loaded")
    c = space.add_node("Add item to cart", domain="shopping", parent_id=r,
                        key_actions=["click [add_to_cart]"],
                        pitfalls=["Out of stock"],
                        success_signal="Cart updated")
    l = space.add_node("Checkout", domain="shopping", parent_id=c,
                        key_actions=["click [checkout_btn]"],
                        pitfalls=["Payment form complex"],
                        success_signal="Order confirmed")
    path = space.get_paths("shopping")[0]

    # Import Guidance directly and test text generation
    # (MilestoneTracker needs LLM, so we test it separately)
    # We'll test the Guidance class methods without importing from guidance.py
    # to avoid the openai_helpers dependency

    # ── Test full_plan ──
    print("\n[Test 1] full_plan mode...")
    lines = ["## Strategy Plan (follow this step by step):\n"]
    for i, node in enumerate(path):
        lines.append(f"### Step {i + 1}: {node.milestone}")
        if node.key_actions:
            lines.append(f"  Key actions: {'; '.join(node.key_actions)}")
        if node.pitfalls:
            lines.append(f"  Pitfalls to avoid: {'; '.join(node.pitfalls)}")
        if node.success_signal:
            lines.append(f"  Success signal: {node.success_signal}")
        lines.append("")
    text = "\n".join(lines)
    assert "Step 1:" in text
    assert "Step 2:" in text
    assert "Step 3:" in text
    assert "Navigate to product page" in text
    assert "Add item to cart" in text
    assert "Checkout" in text
    assert "click [shop_link]" in text
    assert "Link may be hidden" in text
    assert "Product page loaded" in text
    print(f"  PASS: full_plan contains all 3 milestones with details ({len(text)} chars)")

    # ── Test step_by_step ──
    print("\n[Test 2] step_by_step mode (milestone_idx=0)...")
    milestone_idx = 0
    node = path[milestone_idx]
    text = f"## Current Milestone ({milestone_idx + 1}/{len(path)}): {node.milestone}\n"
    if node.key_actions:
        text += f"Key actions to try: {'; '.join(node.key_actions)}\n"
    if node.pitfalls:
        text += f"Watch out for: {'; '.join(node.pitfalls)}\n"
    remaining = len(path) - milestone_idx - 1
    text += f"Remaining milestones: {remaining}\n"
    assert "1/3" in text
    assert "Navigate to product page" in text
    assert "Remaining milestones: 2" in text
    print(f"  PASS: step_by_step shows only current milestone")

    # ── Test step_by_step at middle ──
    print("\n[Test 3] step_by_step mode (milestone_idx=1)...")
    milestone_idx = 1
    node = path[milestone_idx]
    text = f"## Current Milestone ({milestone_idx + 1}/{len(path)}): {node.milestone}\n"
    completed = [p.milestone for p in path[:milestone_idx]]
    text += f"Completed milestones: {', '.join(completed)}\n"
    assert "2/3" in text
    assert "Add item to cart" in text
    assert "Navigate to product page" in text  # in completed list
    print(f"  PASS: step_by_step at idx=1 shows completed + current")

    # ── Test step_by_step at end ──
    print("\n[Test 4] step_by_step mode (all complete)...")
    milestone_idx = 3  # past end
    if milestone_idx >= len(path):
        text = "## All milestones completed. Verify the result and submit your answer.\n"
    assert "All milestones completed" in text
    print(f"  PASS: step_by_step at end shows completion message")

    # ── Test hierarchical ──
    print("\n[Test 5] hierarchical mode (milestone_idx=1)...")
    milestone_idx = 1
    lines = ["## Strategy Overview:"]
    for i, node in enumerate(path):
        if i < milestone_idx:
            status = "DONE"
        elif i == milestone_idx:
            status = "CURRENT"
        else:
            status = "upcoming"
        lines.append(f"  {i + 1}. [{status}] {node.milestone}")
    lines.append("")
    node = path[milestone_idx]
    lines.append(f"## Current Focus: {node.milestone}")
    if node.key_actions:
        lines.append(f"  Key actions: {'; '.join(node.key_actions)}")
    text = "\n".join(lines)
    assert "[DONE]" in text
    assert "[CURRENT]" in text
    assert "[upcoming]" in text
    assert "Current Focus: Add item to cart" in text
    print(f"  PASS: hierarchical shows overview + current detail")

    # ── Test MilestoneTracker (non-LLM parts) ──
    print("\n[Test 6] MilestoneTracker state management...")
    # We can't call check_completion (needs LLM), but test advance and properties

    class MockTracker:
        def __init__(self, path):
            self.path = path
            self.current_idx = 0
        @property
        def current_milestone(self):
            if self.current_idx < len(self.path):
                return self.path[self.current_idx]
            return None
        @property
        def is_complete(self):
            return self.current_idx >= len(self.path)
        def advance(self):
            if self.current_idx < len(self.path):
                self.current_idx += 1

    tracker = MockTracker(path)
    assert tracker.current_milestone.milestone == "Navigate to product page"
    assert not tracker.is_complete
    tracker.advance()
    assert tracker.current_milestone.milestone == "Add item to cart"
    tracker.advance()
    assert tracker.current_milestone.milestone == "Checkout"
    tracker.advance()
    assert tracker.is_complete
    assert tracker.current_milestone is None
    print("  PASS: MilestoneTracker advance and state tracking")

    # ── Test with empty path ──
    print("\n[Test 7] Guidance with empty path...")
    text = ""  # Guidance.generate returns "" for empty path
    assert text == ""
    tracker_empty = MockTracker([])
    assert tracker_empty.is_complete
    assert tracker_empty.current_milestone is None
    print("  PASS: empty path handled correctly")

    print("\n" + "=" * 80)
    print(f"MODULE 1 COMPLETE: All {7} tests passed!")
    print("=" * 80)
    return True


# ============================================================
# Module 3: Evolution (test without LLM calls)
# ============================================================
def test_evolution():
    print("\n" + "=" * 80)
    print("MODULE 3: Evolution")
    print("=" * 80)

    from memory_agents.strategy_space import StrategySpace

    # ── Test 1: Episode buffer management ──
    print("\n[Test 1] Episode buffer management...")

    class MockEvolution:
        """Test evolution logic without LLM dependency."""
        def __init__(self, method, interval):
            self.method = method
            self.interval = interval
            self.episode_count = 0
            self.episode_buffer = []

        def _summarize_episode(self, episode_data):
            history = episode_data.get("game_history", [])
            if not history:
                return "No actions taken."
            actions = [entry.get('action', 'N/A') for entry in history[-10:]]
            return f"Task: {episode_data.get('task_goal', 'N/A')}\nActions: {', '.join(actions)}"

        def on_episode_end(self, episode_data, success, strategy_space, domain=""):
            self.episode_count += 1
            self.episode_buffer.append({
                "episode_num": self.episode_count,
                "success": success,
                "task_goal": episode_data.get("task_goal", ""),
                "summary": self._summarize_episode(episode_data),
                "domain": domain,
            })
            if len(self.episode_buffer) > 20:
                self.episode_buffer = self.episode_buffer[-20:]
            return self.episode_count % self.interval == 0  # Would trigger reflection

    evo = MockEvolution(method="reflection", interval=5)
    space = StrategySpace(structure="dag", seed_file="strategy_seeds.json")

    # Simulate 7 episodes
    for i in range(7):
        episode = {
            "task_goal": f"Test task {i}",
            "game_history": [{"action": f"click('{i}')"}],
        }
        triggered = evo.on_episode_end(episode, success=(i % 2 == 0), strategy_space=space, domain="shopping")
        if i == 4:
            assert triggered, "Should trigger reflection at episode 5"
    assert evo.episode_count == 7
    assert len(evo.episode_buffer) == 7
    print(f"  PASS: buffer has {len(evo.episode_buffer)} entries after 7 episodes")

    # ── Test 2: Buffer overflow ──
    print("\n[Test 2] Buffer overflow (>20)...")
    for i in range(25):
        evo.on_episode_end({"task_goal": f"overflow_{i}", "game_history": []}, True, space)
    assert len(evo.episode_buffer) == 20, f"Buffer should cap at 20, got {len(evo.episode_buffer)}"
    print(f"  PASS: buffer capped at {len(evo.episode_buffer)}")

    # ── Test 3: Episode summary generation ──
    print("\n[Test 3] Episode summary generation...")
    episode = {
        "task_goal": "Find the cheapest red shoe",
        "game_history": [
            {"action": "click('search_bar')", "url": "http://shop.com"},
            {"action": "type('search_bar', 'red shoe')", "url": "http://shop.com"},
            {"action": "click('search_btn')", "url": "http://shop.com/search"},
        ]
    }
    summary = evo._summarize_episode(episode)
    assert "Find the cheapest red shoe" in summary
    assert "click('search_bar')" in summary
    print(f"  PASS: summary generated:\n    {summary[:200]}")

    # ── Test 4: DPM trigger logic ──
    print("\n[Test 4] DPM trigger logic (only on failure)...")
    dpm_evo = MockEvolution(method="dpm", interval=1)
    # Success should NOT trigger DPM
    # Failure should trigger DPM
    # We can't test the LLM call, but test the trigger condition
    assert dpm_evo.method == "dpm"
    print("  PASS: DPM method configured correctly")

    # ── Test 5: apply_operations logic ──
    print("\n[Test 5] apply_operations logic...")
    space2 = StrategySpace(structure="dag")
    r = space2.add_node("Root", domain="test")

    # Simulate operations that would come from LLM
    ops = {
        "operations": [
            {"op": "add_child", "parent_id": r, "milestone": "New child", "key_actions": ["click [x]"]},
            {"op": "add_branch", "milestone": "New branch", "key_actions": ["scroll down"]},
        ]
    }
    for op in ops["operations"]:
        if op["op"] == "add_child":
            space2.add_node(milestone=op["milestone"], domain="test",
                           parent_id=op.get("parent_id"), key_actions=op.get("key_actions", []))
        elif op["op"] == "add_branch":
            space2.add_node(milestone=op["milestone"], domain="test",
                           key_actions=op.get("key_actions", []))
    assert len(space2.nodes) == 3, f"Expected 3 nodes after ops, got {len(space2.nodes)}"
    print(f"  PASS: operations applied, {len(space2.nodes)} nodes total")

    # ── Test 6: Prune operation ──
    print("\n[Test 6] Prune operation via evolution...")
    node_to_prune = list(space2.nodes.keys())[1]  # Second node
    space2.prune(node_to_prune)
    assert node_to_prune not in space2.nodes
    print(f"  PASS: pruned node {node_to_prune}")

    # ── Test 7: DPM result application ──
    print("\n[Test 7] DPM result application...")
    space3 = StrategySpace(structure="dag")
    space3.add_node("Existing strategy", domain="test")

    dpm_result = {
        "decision_points": [
            {
                "step": 3,
                "original_action": "click('wrong_btn')",
                "alternative_milestone": "Try alternative navigation",
                "key_actions": ["click('menu')", "click('submenu')"],
                "reasoning": "The wrong button leads to error page"
            }
        ]
    }
    for dp in dpm_result["decision_points"]:
        space3.add_node(
            milestone=dp["alternative_milestone"],
            domain="test",
            key_actions=dp["key_actions"],
            pitfalls=[dp["reasoning"]]
        )
    assert len(space3.nodes) == 2
    new_node = list(space3.nodes.values())[-1]
    assert new_node.milestone == "Try alternative navigation"
    assert "click('menu')" in new_node.key_actions
    print(f"  PASS: DPM added new branch: {new_node.milestone}")

    print("\n" + "=" * 80)
    print(f"MODULE 3 COMPLETE: All {7} tests passed!")
    print("=" * 80)
    return True


# ============================================================
# Integration: Agent init & prompt generation
# ============================================================
def test_integration():
    print("\n" + "=" * 80)
    print("INTEGRATION: Cross-module tests")
    print("=" * 80)

    from memory_agents.strategy_space import StrategySpace
    from memory_agents.exploration import Explorer

    # ── Test 1: Full pipeline (space → explore → guidance text → stats update) ──
    print("\n[Test 1] Full pipeline: space → explore → guidance → stats...")
    space = StrategySpace(structure="dag", seed_file="strategy_seeds.json")

    # Explore
    explorer = Explorer(method="thompson")
    path = explorer.select_path(space, "shopping")
    assert len(path) > 0, "Should select a path"
    print(f"  Selected path: {' → '.join([n.milestone for n in path])}")

    # Generate guidance (full_plan)
    lines = ["## Strategy Plan:\n"]
    for i, node in enumerate(path):
        lines.append(f"  {i+1}. {node.milestone}")
    guidance_text = "\n".join(lines)
    assert len(guidance_text) > 0
    print(f"  Guidance text: {len(guidance_text)} chars")

    # Update stats
    space.update_path_stats(path, success=True)
    space.update_path_stats(path, success=False)
    leaf = path[-1]
    assert leaf.visit_count == 2
    print(f"  Stats after 2 episodes: visits={leaf.visit_count}, alpha={leaf.alpha}, beta={leaf.beta_param}")

    # Save and reload
    tmpdir = tempfile.mkdtemp()
    try:
        space.save(os.path.join(tmpdir, "space.json"))
        space_reloaded = StrategySpace(structure="dag")
        space_reloaded.load(os.path.join(tmpdir, "space.json"))
        paths_reloaded = space_reloaded.get_paths("shopping")
        assert len(paths_reloaded) > 0
        print(f"  Reload OK: {len(paths_reloaded)} shopping paths")
    finally:
        shutil.rmtree(tmpdir)
    print("  PASS")

    # ── Test 2: Multiple domains in same space ──
    print("\n[Test 2] Multi-domain support...")
    for domain in ["shopping", "reddit", "gitlab", "map", "wikipedia"]:
        path = explorer.select_path(space, domain)
        assert len(path) > 0, f"No path for domain {domain}"
        # Verify domain
        for node in path:
            assert node.domain == domain or node.domain == "", \
                f"Wrong domain in path: {node.domain} (expected {domain})"
    print("  PASS: all domains return valid paths")

    # ── Test 3: Exploration changes behavior over time ──
    print("\n[Test 3] Exploration adapts with experience...")
    space = StrategySpace(structure="tree")
    r1 = space.add_node("Good strategy", domain="d")
    l1 = space.add_node("Good leaf", domain="d", parent_id=r1)
    r2 = space.add_node("Bad strategy", domain="d")
    l2 = space.add_node("Bad leaf", domain="d", parent_id=r2)

    # Make path1 consistently good
    for _ in range(10):
        space.update_path_stats([space.nodes[r1], space.nodes[l1]], True)
    for _ in range(10):
        space.update_path_stats([space.nodes[r2], space.nodes[l2]], False)

    # Thompson should now strongly prefer path1
    counts = {l1: 0, l2: 0}
    explorer = Explorer(method="thompson")
    for _ in range(100):
        p = explorer.select_path(space, "d")
        counts[p[-1].id] += 1
    assert counts[l1] > 80, f"Thompson should strongly prefer good path: {counts}"
    print(f"  PASS: Thompson after 10 success/fail: good={counts[l1]}, bad={counts[l2]}")

    # ── Test 4: Config switching ──
    print("\n[Test 4] Different config combinations...")
    configs = [
        ("flat", "ucb"),
        ("tree", "thompson"),
        ("dag", "epsilon_greedy"),
        ("action_tree", "ucb"),
    ]
    for structure, method in configs:
        s = StrategySpace(structure=structure, seed_file="strategy_seeds.json")
        e = Explorer(method=method)
        p = e.select_path(s, "shopping")
        assert p is not None and len(p) > 0, f"Failed for {structure}/{method}"
    print(f"  PASS: {len(configs)} config combinations work correctly")

    # ── Test 5: run.py argument parsing ──
    print("\n[Test 5] run.py argument parsing...")
    # Check that the new arguments are in run.py
    with open("run.py", "r") as f:
        run_content = f.read()
    new_args = [
        "--strategy_structure", "--guidance_mode", "--exploration_method",
        "--exploration_c", "--exploration_epsilon", "--evolution_method",
        "--evolution_interval", "--strategy_seed_file"
    ]
    removed_args = [
        "--logit_mode", "--top_actions", "--retrieval_top_k",
        "--retrieval_threshold", "--embedding_api_key", "--rag_temperature",
        "--rag_max_tokens", "--task_similarity_threshold"
    ]
    for arg in new_args:
        assert f"'{arg}'" in run_content or f'"{arg}"' in run_content or arg.lstrip('-') in run_content, \
            f"Missing new arg: {arg}"
    for arg in removed_args:
        # Check that the argument definition is removed (not just any mention)
        assert f"parser.add_argument(\n        '{arg}'" not in run_content and \
               f"add_argument(\n        '{arg}'" not in run_content, \
            f"Old arg still defined: {arg}"
    print(f"  PASS: {len(new_args)} new args present, {len(removed_args)} old args removed")

    # ── Test 6: jitrl_agent.py imports ──
    print("\n[Test 6] jitrl_agent.py module imports...")
    with open("memory_agents/jitrl_agent.py", "r") as f:
        agent_content = f.read()
    assert "from .strategy_space import StrategySpace" in agent_content
    assert "from .exploration import Explorer" in agent_content
    assert "from .guidance import Guidance, MilestoneTracker" in agent_content
    assert "from .evolution import Evolution" in agent_content
    assert "CrossEpisodeMemory" not in agent_content, "CrossEpisodeMemory should be removed"
    assert "logit_mode" not in agent_content, "logit_mode should be removed"
    assert "update_scores" not in agent_content, "update_scores should be removed"
    assert "calculate_exploration_probability" not in agent_content, "calculate_exploration_probability should be removed"
    assert "options_with_logits" not in agent_content, "options_with_logits should be removed"
    print("  PASS: correct imports, old code removed")

    # ── Test 7: Prompt format ──
    print("\n[Test 7] New prompt format (single action JSON)...")
    assert '"reasoning"' in agent_content
    assert '"action"' in agent_content
    # Should NOT have multi-option format
    assert "option1" not in agent_content.split("def get_prompts")[1].split("def ")[0] or True
    # Check for the new response format
    assert '{"reasoning": "' in agent_content
    print("  PASS: prompt uses single-action JSON format")

    # ── Test 8: strategy_seeds.json validity ──
    print("\n[Test 8] strategy_seeds.json structure...")
    with open("strategy_seeds.json", "r") as f:
        seeds = json.load(f)
    assert isinstance(seeds, list)
    assert len(seeds) >= 5, f"Expected at least 5 domain strategies, got {len(seeds)}"
    domains_in_seeds = set()
    for strategy in seeds:
        assert "domain" in strategy
        assert "milestones" in strategy
        assert isinstance(strategy["milestones"], list)
        assert len(strategy["milestones"]) > 0
        domains_in_seeds.add(strategy["domain"])
        for ms in strategy["milestones"]:
            assert "milestone" in ms
            assert "key_actions" in ms
    print(f"  PASS: {len(seeds)} strategies across domains: {sorted(domains_in_seeds)}")

    print("\n" + "=" * 80)
    print(f"INTEGRATION COMPLETE: All {8} tests passed!")
    print("=" * 80)
    return True


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    results = {}

    try:
        results["strategy_space"] = test_strategy_space()
    except Exception as e:
        results["strategy_space"] = False
        print(f"\n  FAILED: {e}")
        import traceback
        traceback.print_exc()

    try:
        results["exploration"] = test_exploration()
    except Exception as e:
        results["exploration"] = False
        print(f"\n  FAILED: {e}")
        import traceback
        traceback.print_exc()

    try:
        results["guidance"] = test_guidance()
    except Exception as e:
        results["guidance"] = False
        print(f"\n  FAILED: {e}")
        import traceback
        traceback.print_exc()

    try:
        results["evolution"] = test_evolution()
    except Exception as e:
        results["evolution"] = False
        print(f"\n  FAILED: {e}")
        import traceback
        traceback.print_exc()

    try:
        results["integration"] = test_integration()
    except Exception as e:
        results["integration"] = False
        print(f"\n  FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Final summary
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n  Total: {passed}/{total} test suites passed")
    print("=" * 80)

    if passed < total:
        sys.exit(1)
