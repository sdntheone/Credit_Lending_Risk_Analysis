from flask_wtf import FlaskForm
from wtforms import IntegerField, FloatField, SelectField, SubmitField
from wtforms.validators import Optional, NumberRange, ValidationError

def percent_validator(form, field):
    if field.data is not None:
        if not (0 <= field.data <= 100):
            raise ValidationError('Percentage must be between 0 and 100')

class CreditApprovalForm(FlaskForm):
    last_prod_enq2 = SelectField('Last Product Enquiry 2', choices=[('AL', 'AL'),('CC', 'CC'),
        ('ConsumerLoan', 'ConsumerLoan'),
        ('HL', 'HL'),
        ('OTHERS', 'OTHERS'),
        ('PL', 'PL')], validators=[Optional()], default='AL')
    
    first_prod_enq2 = SelectField('First Product Enquiry 2', choices=[('AL', 'AL'),('CC', 'CC'),
        ('ConsumerLoan', 'ConsumerLoan'),
        ('HL', 'HL'),
        ('OTHERS', 'OTHERS'),
        ('PL', 'PL')], validators=[Optional()], default='AL')

    Total_TL_opened_L6M = IntegerField('Total TL Opened Last 6 Months', validators=[Optional()], default=0)
    Tot_TL_closed_L6M = IntegerField('Total TL Closed Last 6 Months', validators=[Optional()], default=0)
    pct_tl_closed_L6M = FloatField('Percent TL Closed Last 6 Months', validators=[Optional(), percent_validator], default=0.0)
    pct_closed_tl = FloatField('Percent Closed TL', validators=[Optional(), percent_validator], default=0.0)
    pct_tl_open_L12M = FloatField('Percent TL Open Last 12 Months', validators=[Optional(), percent_validator], default=0.0)
    pct_tl_closed_L12M = FloatField('Percent TL Closed Last 12 Months', validators=[Optional(), percent_validator], default=0.0)

    Tot_Missed_Pmnt = IntegerField('Total Missed Payments', validators=[Optional()], default=0)
    CC_TL = IntegerField('CC TL', validators=[Optional()], default=0)
    Consumer_TL = IntegerField('Consumer TL', validators=[Optional()], default=0)
    Gold_TL = IntegerField('Gold TL', validators=[Optional()], default=0)
    Home_TL = IntegerField('Home TL', validators=[Optional()], default=0)
    PL_TL = IntegerField('PL TL', validators=[Optional()], default=0)
    Other_TL = IntegerField('Other TL', validators=[Optional()], default=0)

    Age_Oldest_TL = IntegerField('Age Oldest TL', validators=[Optional()], default=0)
    Age_Newest_TL = IntegerField('Age Newest TL', validators=[Optional()], default=0)
    time_since_recent_payment = IntegerField('Time Since Recent Payment', validators=[Optional()], default=0)

    num_times_delinquent = IntegerField('Number of Times Delinquent', validators=[Optional()], default=0)
    max_recent_level_of_deliq = IntegerField('Max Recent Level of Delinquency', validators=[Optional()], default=0)
    num_deliq_6_12mts = IntegerField('Number of Delinquencies 6-12 Months', validators=[Optional()], default=0)

    num_std = IntegerField('Number of Standard', validators=[Optional()], default=0)
    num_std_6mts = IntegerField('Number of Standard Last 6 Months', validators=[Optional()], default=0)

    recent_level_of_deliq = IntegerField('Recent Level of Delinquency', validators=[Optional()], default=0)
    time_since_recent_enq = IntegerField('Time Since Recent Enquiry', validators=[Optional()], default=0)
    Time_With_Curr_Empr = IntegerField('Time with Current Employer', validators=[Optional()], default=0)

    pct_opened_TLs_L6m_of_L12m = FloatField('Percent Opened TLs Last 6M of Last 12M', validators=[Optional(), percent_validator], default=0.0)
    pct_PL_enq_L6m_of_ever = FloatField('Percent PL Enquiries Last 6M of Ever', validators=[Optional(), percent_validator], default=0.0)
    pct_CC_enq_L6m_of_ever = FloatField('Percent CC Enquiries Last 6M of Ever', validators=[Optional(), percent_validator], default=0.0)

    MARITALSTATUS = SelectField('Marital Status', choices=[('Single', 'Single'), ('Married', 'Married')], validators=[Optional()], default='Single')
    PL_Flag = SelectField('PL Flag', choices=[('1', '1'), ('0', '0')], validators=[Optional()], default='N')
    HL_Flag = SelectField('HL Flag', choices=[('1', '1'), ('0', '0')], validators=[Optional()], default='N')
    GL_Flag = SelectField('GL Flag', choices=[('1', '1'), ('0', '0')], validators=[Optional()], default='N')

    # Newly added fields
    GENDER = SelectField('Gender', choices=[('Male', 'Male'), ('Female', 'Female')], validators=[Optional()], default='Male')
    num_sub_12mts = IntegerField('Number of Subscriptions in 12 Months', validators=[Optional()], default=0)
    num_times_60p_dpd = IntegerField('Number of Times 60+ DPD', validators=[Optional()], default=0)
    NETMONTHLYINCOME = FloatField('Net Monthly Income', validators=[Optional()], default=0.0)
    num_sub = IntegerField('Total Number of Subscriptions', validators=[Optional()], default=0)
    EDUCATION = SelectField('Education', choices=[('12TH', '12TH'),
        ('POST-GRADUATE', 'POST-GRADUATE'),
        ('OTHERS', 'OTHERS'),
        ('GRADUATE', 'GRADUATE'),
        ('SSC', 'SSC'),
        ('PROFESSIONAL', 'PROFESSIONAL'),
        ('UNDER GRADUATE', 'UNDER GRADUATE')], validators=[Optional()], default='Graduate')
    num_lss = IntegerField('Number of Losses', validators=[Optional()], default=0)
    CC_Flag = SelectField('CC Flag', choices=[('1', '1'), ('0', '0')], validators=[Optional()], default='N')
    num_dbt = IntegerField('Number of Debts', validators=[Optional()], default=0)

    submit = SubmitField('Predict')
