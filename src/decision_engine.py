import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("decision_engine")

# ==========================================
# 1. FINANCIAL COST MATRIX (Hypothetical Base Units)
# ==========================================
# These values quantify the trade-off. In a real deployment,
# the finance team provides these exact coefficients.
COST_CARTING_PER_KM = 1.0     # Baseline cheap cost
COST_FTL_PER_KM = 2.5         # Premium cost (dedicated truck)
SLA_BREACH_PENALTY = 500.0    # Cost of unhappy customer, SLA refund, and downstream network chaos

# ==========================================
# 2. DATA PREPARATION & MERGE
# ==========================================
def prepare_decision_data(trip_path, node_path):
    log.info("Loading trip and graph metrics...")

    trip_df = pd.read_csv(trip_path, low_memory=False)
    node_df = pd.read_csv(node_path)

    # Define Target: SLA Breach (Actual Time > 1.2x OSRM Time)
    trip_df['is_sla_breach'] = (trip_df['delay_ratio'] > 1.2).astype(int)

    # Feature Engineering
    trip_df['hour_of_day'] = pd.to_datetime(trip_df['od_start_time']).dt.hour
    trip_df['route_type_encoded'] = trip_df['route_type'].map({'FTL': 1, 'Carting': 0})

    # Merge Graph Centrality Metrics for the Source Hub
    # (A highly central hub under load might necessitate FTL to avoid compounding delays)
    node_metrics = node_df[['node_id', 'betweenness', 'importance_score']]
    trip_df = trip_df.merge(node_metrics, left_on='source_center', right_on='node_id', how='left')

    # Fill any missing graph metrics with 0 (for isolated nodes)
    trip_df['betweenness'] = trip_df['betweenness'].fillna(0)
    trip_df['importance_score'] = trip_df['importance_score'].fillna(0)

    return trip_df.dropna(subset=['route_type_encoded'])

# ==========================================
# 3. MODEL TRAINING
# ==========================================
def train_breach_predictor(df):
    log.info("Training SLA Breach Classifier...")

    features = [
        'segment_osrm_distance',
        'hour_of_day',
        'betweenness',         # Graph position risk
        'importance_score',    # Overall hub vulnerability
        'route_type_encoded'   # 1 for FTL, 0 for Carting
    ]

    X = df[features].copy()
    y = df['is_sla_breach']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # XGBClassifier configured for probability outputs
    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train), # Handle class imbalance
        eval_metric='auc',
        random_state=42
    )

    clf.fit(X_train, y_train)

    # Evaluate
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)
    log.info(f"Model AUC-ROC: {auc:.4f}")
    log.info("\n" + classification_report(y_test, preds))

    return clf, features

# ==========================================
# 4. COUNTERFACTUAL DECISION ENGINE
# ==========================================
def optimize_route_selection(clf, df, features):
    log.info("Running Counterfactual Decision Engine for all trips...")

    # Prepare the feature space
    eval_df = df[features].copy()

    # 1. Simulate probability of breach if EVERYTHING was sent via Carting
    carting_scenario = eval_df.copy()
    carting_scenario['route_type_encoded'] = 0
    p_breach_carting = clf.predict_proba(carting_scenario)[:, 1]

    # 2. Simulate probability of breach if EVERYTHING was sent via FTL
    ftl_scenario = eval_df.copy()
    ftl_scenario['route_type_encoded'] = 1
    p_breach_ftl = clf.predict_proba(ftl_scenario)[:, 1]

    # 3. Calculate Expected Costs
    distances = df['segment_osrm_distance'].values

    expected_cost_carting = (distances * COST_CARTING_PER_KM) + (p_breach_carting * SLA_BREACH_PENALTY)
    expected_cost_ftl = (distances * COST_FTL_PER_KM) + (p_breach_ftl * SLA_BREACH_PENALTY)

    # 4. Make Data-Backed Recommendations
    recommendations = pd.DataFrame({
        'trip_uuid': df['trip_uuid'],
        'source_center': df['source_center'],
        'osrm_distance': distances,
        'p_breach_carting': np.round(p_breach_carting, 3),
        'p_breach_ftl': np.round(p_breach_ftl, 3),
        'exp_cost_carting': expected_cost_carting,
        'exp_cost_ftl': expected_cost_ftl,
        'historical_choice': df['route_type']
    })

    # The engine chooses the route type that minimizes total expected cost
    recommendations['recommended_route_type'] = np.where(
        expected_cost_ftl < expected_cost_carting,
        'FTL',
        'Carting'
    )

    # Calculate potential savings
    recommendations['cost_savings'] = np.abs(expected_cost_carting - expected_cost_ftl)

    return recommendations

# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    trip_path = '/mnt/user-data/outputs/delivery_processed.csv'
    node_path = '/mnt/user-data/outputs/hub_audit_metrics.csv'

    # Run Pipeline
    df = prepare_decision_data(trip_path, node_path)
    clf, feature_cols = train_breach_predictor(df)
    decision_matrix = optimize_route_selection(clf, df, feature_cols)

    # Extract insights for the Strategy Memo (Task 5)
    upgrades = decision_matrix[(decision_matrix['historical_choice'] == 'Carting') & (decision_matrix['recommended_route_type'] == 'FTL')]
    downgrades = decision_matrix[(decision_matrix['historical_choice'] == 'FTL') & (decision_matrix['recommended_route_type'] == 'Carting')]

    log.info("--- STRATEGY INSIGHTS ---")
    log.info(f"Trips historically sent via Carting that SHOULD be FTL (High SLA Risk): {len(upgrades)}")
    log.info(f"Trips historically sent via FTL that COULD be Carting (Wasted Spend): {len(downgrades)}")

    decision_matrix.to_csv('route_decision_framework.csv', index=False)
    log.info("Decision framework exported successfully.")