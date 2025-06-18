import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template
from forms import CreditApprovalForm

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret_key"

# Load the trained model pipeline
model = joblib.load("best_pipeline.joblib")

@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html", title="Home")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    form = CreditApprovalForm()
    message = None  # Initialize message to avoid UnboundLocalError

    if form.validate_on_submit():
        x_new = pd.DataFrame({
            'first_prod_enq2': [form.first_prod_enq2.data],
            'GENDER': [form.GENDER.data],
            'num_sub_12mts': [form.num_sub_12mts.data],
            'last_prod_enq2': [form.last_prod_enq2.data],
            'num_times_60p_dpd': [form.num_times_60p_dpd.data],
            'NETMONTHLYINCOME': [form.NETMONTHLYINCOME.data],
            'num_sub': [form.num_sub.data],
            'EDUCATION': [form.EDUCATION.data],
            'num_lss': [form.num_lss.data],
            'MARITALSTATUS': [form.MARITALSTATUS.data],
            'CC_Flag': [int(form.CC_Flag.data)],
            'Total_TL_opened_L6M': [form.Total_TL_opened_L6M.data],
            'Tot_TL_closed_L6M': [form.Tot_TL_closed_L6M.data],
            'pct_tl_closed_L6M': [form.pct_tl_closed_L6M.data],
            'pct_closed_tl': [form.pct_closed_tl.data],
            'pct_tl_open_L12M': [form.pct_tl_open_L12M.data],
            'pct_tl_closed_L12M': [form.pct_tl_closed_L12M.data],
            'Tot_Missed_Pmnt': [form.Tot_Missed_Pmnt.data],
            'CC_TL': [form.CC_TL.data],
            'Consumer_TL': [form.Consumer_TL.data],
            'Gold_TL': [form.Gold_TL.data],
            'Home_TL': [form.Home_TL.data],
            'PL_TL': [form.PL_TL.data],
            'Other_TL': [form.Other_TL.data],
            'Age_Oldest_TL': [form.Age_Oldest_TL.data],
            'Age_Newest_TL': [form.Age_Newest_TL.data],
            'time_since_recent_payment': [form.time_since_recent_payment.data],
            'num_times_delinquent': [form.num_times_delinquent.data],
            'max_recent_level_of_deliq': [form.max_recent_level_of_deliq.data],
            'num_deliq_6_12mts': [form.num_deliq_6_12mts.data],
            'num_std': [form.num_std.data],
            'num_std_6mts': [form.num_std_6mts.data],
            'recent_level_of_deliq': [form.recent_level_of_deliq.data],
            'time_since_recent_enq': [form.time_since_recent_enq.data],
            'Time_With_Curr_Empr': [form.Time_With_Curr_Empr.data],
            'pct_opened_TLs_L6m_of_L12m': [form.pct_opened_TLs_L6m_of_L12m.data],
            'pct_PL_enq_L6m_of_ever': [form.pct_PL_enq_L6m_of_ever.data],
            'pct_CC_enq_L6m_of_ever': [form.pct_CC_enq_L6m_of_ever.data],
            'PL_Flag': [int(form.PL_Flag.data)],
            'HL_Flag': [int(form.HL_Flag.data)],
            'GL_Flag': [int(form.GL_Flag.data)],
            'num_dbt': [int(form.num_dbt.data)]
        })

        # Ensure numeric fields are clean
        numeric_cols = x_new.select_dtypes(include=[np.number]).columns.tolist()
        x_new[numeric_cols] = x_new[numeric_cols].apply(pd.to_numeric, errors='coerce')
        x_new.fillna(0, inplace=True)

        # Prediction
        label_mapping = {
            0: "P1",
            1: "P2",
            2: "P3",
            3: "P4"
        }

        try:
            pred_class_index = model.predict(x_new)[0]
            probas = model.predict_proba(x_new)[0]
            pred_class_label = label_mapping.get(pred_class_index, f"Unknown ({pred_class_index})")
            confidence = max(probas) * 100
            message = f"🟢 Predicted Segment: {pred_class_label} with {confidence:.2f}% confidence."
        except Exception as e:
            message = f"❌ Prediction failed due to an error: {e}"

    return render_template("predict.html", title="Predict", form=form, output=message)

@app.route("/about")
def about():
    return render_template("about.html", title="About")

if __name__ == "__main__":
    app.run(debug=True)
