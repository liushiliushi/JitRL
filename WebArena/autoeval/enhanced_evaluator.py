"""
Enhanced evaluator that supports multiple evaluation types (string_match, url_match, program_html)
Based on the official evaluator2.py but adapted for the current system
"""
import collections
import html
import json
import time
import urllib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from beartype import beartype
from nltk.tokenize import word_tokenize
from playwright.sync_api import CDPSession, Page

from autoeval.helper_functions import (
    PseudoPage,
    gitlab_get_project_memeber_role,
    llm_fuzzy_match,
    llm_ua_match,
    reddit_get_post_url,
    shopping_get_latest_order_url,
    shopping_get_sku_latest_review_author,
    shopping_get_sku_latest_review_rating,
)

# Simple action and state types for our system
class Action(dict):
    """Action representation compatible with our system"""
    def __init__(self, action_type: str = "", answer: str = "", **kwargs):
        super().__init__(**kwargs)
        self["action_type"] = action_type
        self["answer"] = answer

class StateInfo(dict):
    """State information representation"""
    def __init__(self, state: str = "", **kwargs):
        super().__init__(**kwargs)
        self["state"] = state

Trajectory = List[Union[Action, StateInfo]]


class BaseEvaluator:
    def __init__(self, eval_tag: str = "") -> None:
        self.eval_tag = eval_tag

    @beartype
    def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page | PseudoPage,
        client: CDPSession | None = None,
    ) -> float:
        raise NotImplementedError

    @staticmethod
    def get_last_action(trajectory: Trajectory) -> Action:
        """Get the last action from trajectory"""
        try:
            last_action = trajectory[-1]
            if isinstance(last_action, dict):
                return Action(**last_action)
            return last_action
        except Exception:
            raise ValueError(
                "The last element of trajectory should be an action"
            )

    @staticmethod
    def get_last_state(trajectory: Trajectory) -> StateInfo:
        """Get the second last element (state) from trajectory"""
        try:
            last_state = trajectory[-2]
            if isinstance(last_state, dict):
                return StateInfo(**last_state)
            return last_state
        except Exception:
            raise ValueError(
                "The second last element of trajectory should be a state"
            )


class StringEvaluator(BaseEvaluator):
    """Check whether the answer is correct with:
    exact match: the answer is exactly the same as the reference answer
    must include: each phrase in the reference answer must be included in the answer
    fuzzy match: the answer is similar to the reference answer, using LLM judge
    """

    @staticmethod
    @beartype
    def clean_answer(answer: str) -> str:
        answer = answer.strip()
        if answer.startswith("'") and answer.endswith("'"):
            answer = answer[1:-1]
        elif answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        return answer.lower()

    @staticmethod
    @beartype
    def exact_match(ref: str, pred: str) -> float:
        return float(
            StringEvaluator.clean_answer(pred)
            == StringEvaluator.clean_answer(ref)
        )

    @staticmethod
    @beartype
    def must_include(ref: str, pred: str, tokenize: bool = False) -> float:
        clean_ref = StringEvaluator.clean_answer(ref)
        clean_pred = StringEvaluator.clean_answer(pred)
        # tokenize the answer if the ref is a single word
        # prevent false positive (e.g, 0)
        if (
            tokenize
            and len(clean_ref) == 1
            and len(word_tokenize(clean_ref)) == 1
        ):
            tok_pred = word_tokenize(clean_pred)
            return float(clean_ref in tok_pred)
        else:
            return float(clean_ref in clean_pred)

    @staticmethod
    @beartype
    def fuzzy_match(ref: str, pred: str, intent: str) -> float:
        return llm_fuzzy_match(pred, ref, intent)

    @staticmethod
    @beartype
    def ua_match(ref: str, pred: str, intent: str) -> float:
        return llm_ua_match(pred, ref, intent)

    def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page | PseudoPage | None = None,
        client: CDPSession | None = None,
    ) -> float:
        with open(config_file, "r") as f:
            configs = json.load(f)

        last_action = self.get_last_action(trajectory)
        pred = self.clean_answer(last_action.get("answer", ""))

        score = 1.0
        for approach, value in configs["eval"]["reference_answers"].items():
            match approach:
                case "exact_match":
                    score *= self.exact_match(ref=value, pred=pred)

                case "must_include":
                    assert isinstance(value, list)
                    for must_value in value:
                        score *= self.must_include(
                            ref=must_value,
                            pred=pred,
                            tokenize=(len(value) == 1),
                        )
                case "fuzzy_match":
                    intent = configs["intent"]
                    if value == "N/A":
                        # if the instruction only asks the model to generate N/A when encountering an unachievable task
                        # without more concrete reasons
                        score *= self.exact_match(ref=value, pred=pred)
                        # if the instruction also asks the model to generate the reason why the task is unachievable
                        # this should be the default as it will prevent false positive N/A`
                        if score != 1:
                            score = 1.0 * self.ua_match(
                                intent=configs["intent"],
                                ref=configs["eval"]["string_note"],
                                pred=pred,
                            )
                    else:
                        assert isinstance(value, list)
                        for reference in value:
                            score *= self.fuzzy_match(
                                ref=reference, pred=pred, intent=intent
                            )
        return score


def replace_url_placeholders(url: str) -> str:
    """Replace URL placeholders with actual URLs from environment"""
    import os
    replacements = {
        "__REDDIT__": os.getenv("WA_REDDIT", "http://localhost:9999").rstrip('/'),
        "__SHOPPING__": os.getenv("WA_SHOPPING", "http://localhost:7770").rstrip('/'),
        "__SHOPPING_ADMIN__": os.getenv("WA_SHOPPING_ADMIN", "http://localhost:7780/admin").rstrip('/'),
        "__GITLAB__": os.getenv("WA_GITLAB", "http://localhost:8023").rstrip('/'),
        "__WIKIPEDIA__": os.getenv("WA_WIKIPEDIA", "http://localhost:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing").rstrip('/'),
        "__MAP__": os.getenv("WA_MAP", "http://localhost:3000").rstrip('/'),
    }
    for placeholder, actual_url in replacements.items():
        url = url.replace(placeholder, actual_url)
    return url


class URLEvaluator(BaseEvaluator):
    """Check URL matching"""

    @beartype
    def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page | PseudoPage,
        client: CDPSession | None = None,
    ) -> float:
        with open(config_file, "r") as f:
            configs = json.load(f)

        def clean_url(url: str) -> str:
            url = str(url)
            url = url.rstrip("/")
            # Normalize multiple consecutive slashes to single slash (except in protocol://)
            # Replace // with / but preserve http:// or https://
            import re
            # First, temporarily replace protocol:// with a placeholder
            url = url.replace('http://', 'HTTPPROTOCOL')
            url = url.replace('https://', 'HTTPSPROTOCOL')
            # Then replace all consecutive slashes with single slash
            url = re.sub(r'/+', '/', url)
            # Restore the protocol
            url = url.replace('HTTPPROTOCOL', 'http://')
            url = url.replace('HTTPSPROTOCOL', 'https://')
            return url

        def parse_url(url: str) -> tuple[str, dict[str, list[str]]]:
            """Parse a URL into its base, path, and query components."""
            parsed_url = urllib.parse.urlparse(url)
            base_path = parsed_url.netloc + parsed_url.path
            query = urllib.parse.parse_qs(parsed_url.query)
            return base_path, query

        def parse_urls(
            urls: list[str],
        ) -> tuple[list[str], dict[str, set[str]]]:
            """Parse a list of URLs."""
            base_paths = []
            queries = collections.defaultdict(set)
            for url in urls:
                base_path, query = parse_url(url)
                base_paths.append(base_path)
                for k, v in query.items():
                    queries[k].update(v)
            return base_paths, queries

        pred = clean_url(page.url)
        ref_urls = configs["eval"]["reference_url"].split(" |OR| ")
        # Replace URL placeholders (e.g., __SHOPPING__) with actual URLs
        ref_urls = [clean_url(replace_url_placeholders(url)) for url in ref_urls]
        matching_rule = configs["eval"].get("url_note", "GOLD in PRED")

        # Print URL matching information
        print("\n" + "="*80)
        print("URL Matching Evaluation")
        print("="*80)
        print(f"Current URL (pred):")
        print(f"  {pred}")
        print(f"\nGround Truth URL(s) (ref):")
        for i, ref_url in enumerate(ref_urls, 1):
            print(f"  {i}. {ref_url}")
        print(f"\nMatching Rule: {matching_rule}")
        print("="*80 + "\n")
        
        if matching_rule == "GOLD in PRED":
            ref_base_paths, ref_queries = parse_urls(ref_urls)
            pred_base_paths, pred_query = parse_url(pred)

            print("URL Parsing Details:")
            print(f"  Reference base paths: {ref_base_paths}")
            print(f"  Reference queries: {dict(ref_queries)}")
            print(f"  Predicted base path: {pred_base_paths}")
            print(f"  Predicted query: {dict(pred_query)}")
            print()

            base_score = float(
                any(
                    [
                        ref_base_path in pred_base_paths
                        for ref_base_path in ref_base_paths
                    ]
                )
            )
            print(f"Base Path Match Score: {base_score}")

            query_score = 1.0
            for k, possible_values in ref_queries.items():
                k_match = float(
                    any(
                        possible_ref_value in pred_query.get(k, [])
                        for possible_ref_value in possible_values
                    )
                )
                print(f"  Query parameter '{k}' match: {k_match}")
                print(f"    Expected values: {possible_values}")
                print(f"    Actual values: {pred_query.get(k, [])}")
                query_score *= k_match

            print(f"Query Match Score: {query_score}")
            score = base_score * query_score

            print(f"\nFinal URL Match Score: {score}")
            print(f"Result: {'✓ PASS' if score > 0.5 else '✗ FAIL'}")
            print("="*80 + "\n")

        else:
            raise ValueError(f"Unknown matching rule: {matching_rule}")

        return score


class HTMLContentEvaluator(BaseEvaluator):
    """Check whether the contents appear in the page"""

    @beartype
    def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page | PseudoPage,
        client: CDPSession | None = None,
    ) -> float:
        with open(config_file, "r") as f:
            configs = json.load(f)

        targets = configs["eval"]["program_html"]

        print("\n" + "="*80)
        print("HTML Content Evaluation (program_html)")
        print("="*80)
        print(f"Total targets to check: {len(targets)}")
        print()

        score = 1.0
        checkpoint_results = []  # Track each checkpoint result
        for i, target in enumerate(targets, 1):
            print(f"--- Checkpoint {i}/{len(targets)} ---")
            checkpoint_passed = False  # Track if this checkpoint passed
            target_score = 1.0  # Track score for this target

            target_url: str = target["url"]  # which url to check
            if target_url.startswith("func"):
                func = target_url.split("func:")[1]
                func = func.replace("__last_url__", page.url)
                target_url = eval(func)

            # Replace URL placeholders
            target_url = replace_url_placeholders(target_url)

            locator: str = target["locator"]  # js element locator

            print(f"Target URL: {target_url}")
            print(f"Locator: {locator}")

            # navigate to that url
            current_page_url = page.url if hasattr(page, 'url') else ""
            # Skip navigation if target URL is same as current URL (avoid page refresh)
            if target_url != "last" and target_url != current_page_url:
                print(f"Navigating to: {target_url}")
                try:
                    page.goto(target_url, wait_until="networkidle", timeout=10000)
                    print(f"✓ Navigation successful")
                except Exception as e:
                    print(f"✗ Navigation failed: {e}")
                    score = 0.0
                    continue
                time.sleep(1)  # Brief wait for JS to execute
            elif target_url == current_page_url:
                print(f"Already on target URL: {target_url} (skipping navigation)")
            else:
                # url="last" means use the current page
                # Only navigate if this is PseudoPage (post-hoc evaluation)
                # For real Playwright Page (live evaluation), use current state directly
                is_pseudo_page = isinstance(page, PseudoPage)

                if is_pseudo_page:
                    # Post-hoc evaluation: need to navigate to saved URL
                    if hasattr(page, 'url') and page.url and page.url != "about:blank":
                        print(f"Navigating to saved URL: {page.url}")
                        try:
                            page.goto(page.url, wait_until="networkidle", timeout=10000)
                            print(f"✓ Navigation to saved URL successful")
                        except Exception as e:
                            print(f"✗ Navigation to saved URL failed: {e}")
                            score = 0.0
                            continue
                        time.sleep(1)
                    else:
                        print(f"Warning: url='last' but no valid URL saved in trajectory")
                else:
                    # Live evaluation: use current page state directly (no navigation)
                    current_url = page.url if hasattr(page, 'url') else 'unknown'
                    print(f"Using current page state (URL: {current_url})")

            # empty, use the full page
            if not locator.strip():
                selected_element = page.content()
                print(f"Using full page content (length: {len(selected_element)} chars)")
            # use JS to select the element
            elif locator.startswith("document.") or locator.startswith(
                "[...document."
            ):
                if "prep_actions" in target:
                    print(f"Executing prep actions...")
                    try:
                        for prep_action in target["prep_actions"]:
                            page.evaluate(f"() => {prep_action}")
                    except Exception as e:
                        print(f"Prep action error: {e}")
                try:
                    print(f"Evaluating JavaScript: {locator}")
                    selected_element = str(page.evaluate(f"() => {locator}"))
                    if not selected_element:
                        selected_element = ""
                    print(f"✓ JavaScript result: '{selected_element}'")
                except Exception as e:
                    # the page is wrong, return empty
                    selected_element = ""
                    print(f"✗ JavaScript evaluation failed: {e}")
            # run program to call API
            elif locator.startswith("func:"):  # a helper function
                func = locator.split("func:")[1]
                func = func.replace("__page__", "page")
                print(f"Executing helper function: {func}")
                selected_element = eval(func)
                print(f"Function result: {selected_element}")
            else:
                raise ValueError(f"Unknown locator: {locator}")

            selected_element = html.unescape(selected_element)

            # Generate checkpoint description
            checkpoint_description = ""
            if "required_contents" in target:
                if "exact_match" in target["required_contents"]:
                    expected_value = target["required_contents"]["exact_match"]
                    checkpoint_description = f"Element matches '{expected_value}'"
                elif "must_include" in target["required_contents"]:
                    expected_values = target["required_contents"]["must_include"]
                    if len(expected_values) == 1:
                        checkpoint_description = f"Element contains '{expected_values[0]}'"
                    else:
                        checkpoint_description = f"Element contains required values"
                else:
                    checkpoint_description = f"Check element content"
            else:
                checkpoint_description = f"Check element at {locator[:50]}..."

            print(f"\nChecking required contents...")
            if "exact_match" in target["required_contents"]:
                required_contents = target["required_contents"]["exact_match"]
                print(f"  Mode: exact_match")
                print(f"  Expected: '{required_contents}'")
                print(f"  Actual: '{selected_element}'")
                cur_score = StringEvaluator.exact_match(
                    ref=required_contents, pred=selected_element
                )
                print(f"  Match: {cur_score > 0} (score: {cur_score})")
                target_score *= float(cur_score)
                score *= float(cur_score)
            elif "must_include" in target["required_contents"]:
                required_contents = target["required_contents"]["must_include"]
                assert isinstance(required_contents, list)
                # print(f"  Mode: must_include")
                # print(f"  Actual content: '{selected_element}'")
                for j, content in enumerate(required_contents, 1):
                    content_or = content.split(" |OR| ")
                    print(f"  Required content {j}: {content_or}")
                    cur_score = any(
                        [
                            StringEvaluator.must_include(
                                ref=content,
                                pred=selected_element,
                                tokenize=False,
                            )
                            for content in content_or
                        ]
                    )
                    print(f"    Match: {cur_score} (score: {1.0 if cur_score else 0.0})")
                    target_score *= float(cur_score)
                    score *= float(cur_score)
            else:
                raise ValueError(
                    f"Unknown required_contents: {target['required_contents'].keys()}"
                )

            # Record checkpoint result
            checkpoint_passed = (target_score > 0.5)
            checkpoint_results.append({
                "checkpoint": i,
                "description": checkpoint_description,
                "passed": checkpoint_passed,
                "score": target_score
            })

            print(f"Cumulative score after checkpoint {i}: {score}\n")

        print(f"{'='*80}")
        print(f"Final HTML Content Evaluation Score: {score}")
        print(f"Result: {'✓ PASS' if score > 0.5 else '✗ FAIL'}")
        print(f"{'='*80}")

        # Print checkpoint summary
        print("\n" + "="*80)
        print("CHECKPOINT SUMMARY")
        print("="*80)
        passed_count = sum(1 for r in checkpoint_results if r["passed"])
        total_count = len(checkpoint_results)
        print(f"Total: {passed_count}/{total_count} checkpoints passed\n")

        for result in checkpoint_results:
            status = "✓ PASS" if result["passed"] else "✗ FAIL"
            print(f"Checkpoint {result['checkpoint']}: {status}")
            print(f"  Description: {result['description']}")
            print(f"  Score: {result['score']}")
            print()

        print(f"{'='*80}\n")
        return score


class EvaluatorComb:
    """Combination of multiple evaluators"""
    def __init__(self, evaluators: List[BaseEvaluator]) -> None:
        self.evaluators = evaluators

    @beartype
    def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page | PseudoPage,
        client: CDPSession | None = None,
    ) -> float:
        score = 1.0
        for evaluator in self.evaluators:
            cur_score = evaluator(trajectory, config_file, page, client)
            score *= cur_score
        return score


@beartype
def evaluator_router(config_file: Path | str) -> EvaluatorComb:
    """Router to get the evaluator class"""
    with open(config_file, "r") as f:
        configs = json.load(f)

    eval_types = configs["eval"]["eval_types"]
    evaluators: List[BaseEvaluator] = []
    for eval_type in eval_types:
        match eval_type:
            case "string_match":
                evaluators.append(StringEvaluator())
            case "url_match":
                evaluators.append(URLEvaluator())
            case "program_html":
                evaluators.append(HTMLContentEvaluator())
            case _:
                raise ValueError(f"eval_type {eval_type} is not supported")

    return EvaluatorComb(evaluators)


def create_trajectory_from_info(traj_info: Dict[str, Any]) -> Trajectory:
    """Create a trajectory from trajectory info"""
    trajectory = []
    
    # Add states if available
    if "states" in traj_info and traj_info["states"]:
        for state_data in traj_info["states"]:
            state = StateInfo(state=state_data.get("state", ""))
            trajectory.append(state)
    
    # Add final action with response
    final_action = Action(
        action_type="send_msg_to_user",
        answer=traj_info.get("response", "")
    )
    trajectory.append(final_action)
    
    return trajectory


def create_pseudo_page_from_info(traj_info: Dict[str, Any], page: Page = None) -> PseudoPage:
    """Create a PseudoPage from trajectory info"""
    # Try to get the final URL from the trajectory
    final_url = "about:blank"  # default

    # Get the last state to determine the page
    if "states" in traj_info and traj_info["states"]:
        last_state = traj_info["states"][-1]

        # First, try to get URL directly from the saved state (if available)
        if "url" in last_state and last_state["url"]:
            final_url = last_state["url"]
            print(f"[URL] Retrieved from saved state: {final_url}")
        else:
            # Fallback: infer URL from state content
            state_content = last_state.get("state", "")

            # Extract page title from RootWebArea
            # Format: "RootWebArea 'Page Title', focused"
            import re
            title_match = re.search(r"RootWebArea '([^']+)'", state_content)
            page_title = title_match.group(1) if title_match else ""

            print(f"[URL Inference] Page title: '{page_title}'")

            # Infer URL from page title and content
            # Common shopping site patterns
            if "Contact Us" in page_title:
                final_url = "__SHOPPING__/contact/index/"
                print(f"[URL Inference] Detected Contact Us page")
            elif "Living Room Furniture" in state_content:
                final_url = "__SHOPPING__/home-kitchen/furniture/living-room-furniture.html"
                # Check if sorting by price is applied
                if "Sort By" in state_content and "Price" in state_content:
                    final_url += "?product_list_order=price"
                    # Check if descending order is set
                    if "Set Ascending Direction" in state_content:
                        final_url += "&product_list_dir=desc"
                    else:
                        final_url += "&product_list_dir=asc"
                print(f"[URL Inference] Detected Living Room Furniture page with sorting")
            elif "One Stop Market" in page_title and "Product Showcases" in state_content:
                final_url = "__SHOPPING__"
                print(f"[URL Inference] Detected shopping home page")
            # Add more patterns as needed
            else:
                # Try to extract URL from state content as fallback
                if "url:" in state_content.lower():
                    lines = state_content.split("\n")
                    for line in lines:
                        if "url:" in line.lower():
                            final_url = line.split(":", 1)[1].strip()
                            print(f"[URL Inference] Extracted URL from state content")
                            break

            print(f"[URL Inference] Final inferred URL: {final_url}")
    return PseudoPage(page, final_url)
