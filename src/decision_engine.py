import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("decision_engine")

COST_CARTING_PER_KM = 1.0
COST_FTL_PER_KM = 2.5
SLA_BREACH_PENALTY = 500.0

def prepare_decision_data(trip_path, node_path):
    log.info("Loading trip and graph metrics...")

    trip_df = pd.read_csv(trip_path, low_memory=False)
    node_df = pd.read_csv(node_path)

    trip_df['is_sla_breach'] = (trip_df['delay_ratio'] > 1.2).astype(int)
    trip_df['hour_of_day'] = pd.to_datetime(trip_df['od_start_time']).dt.hour
    trip_df['route_type_encoded'] = trip_df['route_type'].map({'FTL': 1, 'Carting': 0})

    node_metrics = node_df[['node_id', 'betweenness', 'importance_score']]
    trip_df = trip_df.merge(node_metrics, left_on='source_center', right_on='node_id', how='left')

    trip_df['betweenness'] = trip_df['betweenness'].fillna(0)
    trip_df['importance_score'] = trip_df['importance_score'].fillna(0)

    return trip_df.dropna(subset=['route_type_encoded'])


def train_breach_predictor(df):
    log.info("Training SLA Breach Classifier...")

    features = [
        'segment_osrm_distance',
        'hour_of_day',
        'betweenness',
        'importance_score',
        'route_type_encoded'
    ]

    X = df[features].copy()
    y = df['is_sla_breach']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
        eval_metric='auc',
        random_state=42
    )

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)
    log.info(f"Model AUC-ROC: {auc:.4f}")
    log.info("\n" + classification_report(y_test, preds))

    return clf, features

def optimize_route_selection(clf, df, features):
    log.info("Running Counterfactual Decision Engine for all trips...")

    eval_df = df[features].copy()

    carting_scenario = eval_df.copy()
    carting_scenario['route_type_encoded'] = 0
    p_breach_carting = clf.predict_proba(carting_scenario)[:, 1]

    ftl_scenario = eval_df.copy()
    ftl_scenario['route_type_encoded'] = 1
    p_breach_ftl = clf.predict_proba(ftl_scenario)[:, 1]

    distances = df['segment_osrm_distance'].values

    expected_cost_carting = (distances * COST_CARTING_PER_KM) + (p_breach_carting * SLA_BREACH_PENALTY)
    expected_cost_ftl = (distances * COST_FTL_PER_KM) + (p_breach_ftl * SLA_BREACH_PENALTY)

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

    recommendations['recommended_route_type'] = np.where(
        expected_cost_ftl < expected_cost_carting,
        'FTL',
        'Carting'
    )

    recommendations['cost_savings'] = np.abs(expected_cost_carting - expected_cost_ftl)

    return recommendations

if __name__ == "__main__":
    trip_path = '/mnt/user-data/outputs/delivery_processed.csv'
    node_path = '/mnt/user-data/outputs/hub_audit_metrics.csv'

    df = prepare_decision_data(trip_path, node_path)
    clf, feature_cols = train_breach_predictor(df)
    decision_matrix = optimize_route_selection(clf, df, feature_cols)

    upgrades = decision_matrix[(decision_matrix['historical_choice'] == 'Carting') & (decision_matrix['recommended_route_type'] == 'FTL')]
    downgrades = decision_matrix[(decision_matrix['historical_choice'] == 'FTL') & (decision_matrix['recommended_route_type'] == 'Carting')]

    log.info("--- STRATEGY INSIGHTS ---")
    log.info(f"Trips historically sent via Carting that SHOULD be FTL (High SLA Risk): {len(upgrades)}")
    log.info(f"Trips historically sent via FTL that COULD be Carting (Wasted Spend): {len(downgrades)}")

    decision_matrix.to_csv('route_decision_framework.csv', index=False)
    log.info("Decision framework exported successfully.")
