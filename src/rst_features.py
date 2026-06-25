import numpy as np

# Try to import isanlp_rst, with robust mock fallback if not available
try:
    from isanlp_rst.parser import Parser as IsanlpRstParser
    ISANLP_AVAILABLE = True
except ImportError:
    ISANLP_AVAILABLE = False
    print("Warning: 'isanlp_rst' not found. A rule-based Mock RST Parser will be used as fallback.")

class MockNode:
    """
    Mock node mimicking the structure of an isanlp_rst tree node.
    """
    def __init__(self, id, start, end, text="", relation="span", nuclearity="NS", left=None, right=None):
        self.id = id
        self.start = start
        self.end = end
        self.text = text
        self.relation = relation
        self.nuclearity = nuclearity
        self.left = left
        self.right = right

class DiscourseParserWrapper:
    """
    Wrapper that manages isanlp_rst_v3 initialization or falls back to Mock RST parsing.
    """
    def __init__(self, device="cuda"):
        self.use_mock = not ISANLP_AVAILABLE
        if not self.use_mock:
            try:
                # cuda_device: 0 represents the first GPU (RTX 4060)
                cuda_device = 0 if device == "cuda" else -1
                print(f"Initializing isanlp_rst_v3 Parser on CUDA device {cuda_device}...")
                self.parser = IsanlpRstParser(
                    hf_model_name='tchewik/isanlp_rst_v3',
                    hf_model_version='gumrrg',
                    cuda_device=cuda_device
                )
            except Exception as e:
                print(f"Error initializing isanlp_rst: {e}. Falling back to Mock parser.")
                self.use_mock = True

    def parse(self, text, sentence_boundaries=None):
        """
        Parses text and returns the root node of the RST tree.
        """
        if self.use_mock or not text.strip():
            return self._parse_mock(text, sentence_boundaries)
        try:
            res = self.parser(text)
            if 'rst' in res and len(res['rst']) > 0:
                return res['rst'][0]
            else:
                return self._parse_mock(text, sentence_boundaries)
        except Exception as e:
            print(f"Error during isanlp_rst parsing: {e}. Using Mock parser.")
            return self._parse_mock(text, sentence_boundaries)

    def _parse_mock(self, text, sentence_boundaries):
        """
        Builds a mock hierarchical binary tree where the first sentence is the
        central Nucleus and subsequent sentences are nested Satellite elaborations.
        """
        if not sentence_boundaries:
            return MockNode(0, 0, len(text), text)
            
        # Recursive function to build a mock binary tree from sentences
        def build_tree(sents, start_id=0):
            if not sents:
                return None
            if len(sents) == 1:
                s = sents[0]
                return MockNode(start_id, s["start_char"], s["end_char"], s["text"])
                
            # Split: left is the first sentence (Nucleus), right contains the remaining (Satellite)
            left_node = MockNode(start_id + 1, sents[0]["start_char"], sents[0]["end_char"], sents[0]["text"])
            
            # Subtree for remaining sentences
            right_sents = sents[1:]
            right_node = build_tree(right_sents, start_id + 2)
            
            # Combine them
            start = sents[0]["start_char"]
            end = sents[-1]["end_char"]
            combined_text = text[start:end]
            
            # The relation is 'elaboration' and nuclearity is 'NS' (left Nucleus, right Satellite)
            return MockNode(
                id=start_id,
                start=start,
                end=end,
                text=combined_text,
                relation="elaboration",
                nuclearity="NS",
                left=left_node,
                right=right_node
            )
            
        return build_tree(sentence_boundaries)

class RSTFeatureExtractor:
    """
    Extracts sentence-level, passage-level, and relative features from an RST tree.
    """
    def __init__(self):
        pass
        
    def _flatten_tree(self, node, depth=0, parent_relation="root", parent_nuclearity="root"):
        """
        Traverses the RST tree and returns a list of flattened node dictionaries.
        """
        if node is None:
            return []
            
        is_leaf = (node.left is None) and (node.right is None)
        
        # Current node data
        flat_nodes = [{
            "id": node.id,
            "start": node.start,
            "end": node.end,
            "text": getattr(node, "text", ""),
            "relation": parent_relation,
            "nuclearity": parent_nuclearity,
            "depth": depth,
            "is_leaf": is_leaf
        }]
        
        if not is_leaf:
            # Sibling nuclearity parsing
            nuc = getattr(node, "nuclearity", "NS")
            if nuc == "NS":
                left_nuc, right_nuc = "Nucleus", "Satellite"
            elif nuc == "SN":
                left_nuc, right_nuc = "Satellite", "Nucleus"
            elif nuc == "NN":
                left_nuc, right_nuc = "Nucleus", "Nucleus"
            else:
                left_nuc, right_nuc = "Nucleus", "Satellite"
                
            left_rel = node.relation if left_nuc == "Satellite" else "span"
            right_rel = node.relation if right_nuc == "Satellite" else "span"
            if nuc == "NN":
                left_rel = node.relation
                right_rel = node.relation
                
            flat_nodes.extend(self._flatten_tree(node.left, depth + 1, left_rel, left_nuc))
            flat_nodes.extend(self._flatten_tree(node.right, depth + 1, right_rel, right_nuc))
            
        return flat_nodes

    def extract_rst_features(self, root_node, sentence_boundaries):
        """
        Maps sentences to tree EDUs and computes multi-level RST features:
        - Sentence-level (nuclearity density, average depth, root presence, relation counts)
        - Passage-level baseline
        - Relative features (sentence compared to passage baseline)
        """
        if root_node is None or not sentence_boundaries:
            return [{} for _ in sentence_boundaries]
            
        flat_nodes = self._flatten_tree(root_node)
        leaf_edus = [n for n in flat_nodes if n["is_leaf"]]
        
        # Passage-level baselines
        total_nuclei = sum(1 for n in leaf_edus if n["nuclearity"] == "Nucleus")
        total_satellites = sum(1 for n in leaf_edus if n["nuclearity"] == "Satellite")
        max_depth = max(n["depth"] for n in leaf_edus) if leaf_edus else 1
        
        # List of all unique relations in passage
        relations_list = ["elaboration", "attribution", "background", "cause", "result", "contrast", "joint"]
        
        sentence_features = []
        
        for sent in sentence_boundaries:
            s_start = sent["start_char"]
            s_end = sent["end_char"]
            
            # Map EDUs to sentence: non-empty intersection
            sent_edus = [e for e in leaf_edus if max(s_start, e["start"]) < min(s_end, e["end"])]
            
            if not sent_edus:
                # Default empty features
                features = {
                    "rst_edu_count": 0.0, "rst_n_count": 0.0, "rst_s_count": 0.0, "rst_n_ratio": 0.5,
                    "rst_mean_depth": float(max_depth), "rst_is_root": 0.0,
                    "psg_rst_max_depth": float(max_depth), "psg_rst_n_count": float(total_nuclei),
                    "rel_rst_depth_ratio": 1.0, "rel_rst_n_ratio": 0.0
                }
                for r in relations_list:
                    features[f"rst_rel_{r}_count"] = 0.0
                sentence_features.append(features)
                continue
                
            # Sentence level stats
            edu_count = len(sent_edus)
            n_count = sum(1 for e in sent_edus if e["nuclearity"] == "Nucleus")
            s_count = sum(1 for e in sent_edus if e["nuclearity"] == "Satellite")
            n_ratio = n_count / max(1, n_count + s_count)
            
            mean_depth = np.mean([e["depth"] for e in sent_edus])
            is_root = 1.0 if any(e["depth"] <= 1 for e in sent_edus) else 0.0
            
            # Relation counts
            rel_counts = {r: 0.0 for r in relations_list}
            for e in sent_edus:
                rel = e["relation"].lower()
                if rel in rel_counts:
                    rel_counts[rel] += 1.0
                elif "cause" in rel or "result" in rel:
                    rel_counts["cause"] += 1.0
                elif "elaboration" in rel:
                    rel_counts["elaboration"] += 1.0
                    
            # Relative features
            rel_depth_ratio = mean_depth / max(1, max_depth)
            rel_n_ratio = n_count / max(1, total_nuclei)
            
            features = {
                # Sentence level
                "rst_edu_count": float(edu_count),
                "rst_n_count": float(n_count),
                "rst_s_count": float(s_count),
                "rst_n_ratio": n_ratio,
                "rst_mean_depth": float(mean_depth),
                "rst_is_root": is_root,
                # Passage level baselines
                "psg_rst_max_depth": float(max_depth),
                "psg_rst_n_count": float(total_nuclei),
                "psg_rst_s_count": float(total_satellites),
                # Relative
                "rel_rst_depth_ratio": rel_depth_ratio,
                "rel_rst_n_ratio": rel_n_ratio
            }
            
            # Add relation counts
            for r in relations_list:
                features[f"rst_rel_{r}_count"] = rel_counts[r]
                
            sentence_features.append(features)
            
        return sentence_features
