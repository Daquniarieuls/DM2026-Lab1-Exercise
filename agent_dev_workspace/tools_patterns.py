import tempfile
import os
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str,
        algorithm: str = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Mine frequent itemsets/patterns for a specific category's text data.

        Args:
            category_name: Target category label to mine patterns for.
            filtering_method: Vocabulary filtering approach ('variance', 'tfidf', or 'term_frequency').
            algorithm: Mining algorithm to run ('fpgrowth', 'topk', or 'maxfpgrowth').
            min_sup: Minimum support threshold (required for 'fpgrowth' and 'maxfpgrowth').
            k: Top-k patterns count (required for 'topk').
        """
        # 1. Argument validation with {"error": "..."} responses
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Call load_dataset_tool first."}

        valid_filtering = ["variance", "tfidf", "term_frequency"]
        if filtering_method not in valid_filtering:
            return {"error": f"Invalid filtering_method '{filtering_method}'. Must be one of {valid_filtering}."}

        valid_algorithms = ["fpgrowth", "topk", "maxfpgrowth"]
        if algorithm not in valid_algorithms:
            return {"error": f"Invalid algorithm '{algorithm}'. Must be one of {valid_algorithms}."}

        if algorithm in ["fpgrowth", "maxfpgrowth"]:
            if min_sup is None or min_sup <= 0:
                return {"error": f"min_sup (positive integer) is required for algorithm '{algorithm}'."}
        elif algorithm == "topk":
            if k is None or k <= 0:
                return {"error": "k (positive integer) is required for algorithm 'topk'."}

        # Filter dataframe by category
        df = session.dataframe
        if "category_name" in df.columns:
            cat_mask = df["category_name"] == category_name
        elif "category" in df.columns:
            cat_mask = df["category"] == category_name
        else:
            return {"error": "Dataframe must contain 'category_name' or 'category' column."}

        cat_df = df[cat_mask]
        if cat_df.empty:
            return {"error": f"No documents found for category_name '{category_name}'."}

        texts = cat_df["text"].fillna("").tolist()

        # Step 1: Per-category local CountVectorizer with English stop words
        vectorizer = CountVectorizer(stop_words="english")
        dtm_mat = vectorizer.fit_transform(texts)
        feature_names = np.array(vectorizer.get_feature_names_out())

        if len(feature_names) == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_patterns": 0,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
            return summary

        # Step 2: Vocabulary filtering
        dtm_dense = dtm_mat.toarray()
        if filtering_method == "variance":
            scores = np.var(dtm_dense, axis=0)
            p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
            keep_mask = (scores > p5) & (scores < p95)
        elif filtering_method == "term_frequency":
            scores = np.sum(dtm_dense, axis=0)
            p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
            keep_mask = (scores > p5) & (scores < p95)
        elif filtering_method == "tfidf":
            tfidf_transformer = TfidfTransformer()
            tfidf_mat = tfidf_transformer.fit_transform(dtm_mat)
            scores = np.mean(tfidf_mat.toarray(), axis=0)
            p20 = np.percentile(scores, 20)
            keep_mask = scores >= p20

        kept_features = feature_names[keep_mask]
        kept_dtm = dtm_dense[:, keep_mask]

        # Step 3: Handle 0 surviving terms
        if len(kept_features) == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_patterns": 0,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
            return summary

        # Step 4: Convert to Transactional Database and Mine using PAMI
        transaction_df = pd.DataFrame(kept_dtm, columns=kept_features)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "transactions.txt")
            output_file = os.path.join(tmpdir, "output_patterns.txt")

            from PAMI.extras.convert import DF2DB
            conv = DF2DB.DF2DB(transaction_df)
            conv.convert2TransactionalDatabase(input_file, '>=', 1)

            if algorithm == "fpgrowth":
                from PAMI.frequentPattern.basic import FPGrowth
                miner = FPGrowth.FPGrowth(input_file, minSup=min_sup, sep="\t")
            elif algorithm == "topk":
                from PAMI.frequentPattern.topk import FAE
                miner = FAE.FAE(input_file, k=k, sep="\t")
            elif algorithm == "maxfpgrowth":
                from PAMI.frequentPattern.maximal import MaxFPGrowth
                miner = MaxFPGrowth.MaxFPGrowth(input_file, minSup=min_sup, sep="\t")

            miner.mine()

            # Retrieve patterns DataFrame directly from miner
            patterns_df = miner.getPatternsAsDataFrame()

            patterns = []
            if patterns_df is not None and not patterns_df.empty:
                col_itemset = patterns_df.columns[0]
                col_support = patterns_df.columns[1]
                for _, row in patterns_df.iterrows():
                    raw_pat = str(row[col_itemset]).strip()
                    sup_val = float(row[col_support])
                    itemset = raw_pat.split()
                    patterns.append({"pattern": itemset, "support": sup_val})

        # Sort patterns by support descending, then by pattern length/alphabetical
        patterns.sort(key=lambda p: (-p["support"], len(p["pattern"]), " ".join(p["pattern"])))

        result_id = session.next_result_id("patterns")
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns

        # Create full_report DataFrame for store_result
        if patterns:
            full_df = pd.DataFrame([
                {"pattern": ", ".join(p["pattern"]), "support": p["support"]}
                for p in patterns
            ])
        else:
            full_df = pd.DataFrame(columns=["pattern", "support"])

        # Truncate summary pattern list to top 30 sorted patterns if larger
        summary_patterns = patterns[:30]
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_patterns": len(patterns),
            "patterns": summary_patterns
        }
        if len(patterns) > 30:
            summary["note"] = f"summary truncated to top 30 patterns out of {len(patterns)}"

        session.store_result(
            tool_name="mine_patterns_tool",
            args={
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k
            },
            summary=summary,
            full_report=full_df
        )

        return summary

    return [mine_patterns_tool]
