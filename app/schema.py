from pydantic import BaseModel, Field


class PredictionInput(BaseModel):

    enq_L3m: float = Field(..., ge=0, description="Number of enquiries in last 3 months", example=2)

    Age_Oldest_TL: int = Field(..., ge=0, description="Age of oldest tradeline in months", example=120)

    num_times_delinquent: int = Field(..., ge=0, description="Total number of delinquent events", example=1)

    pct_PL_enq_L6m_of_ever: float = Field(..., ge=0, le=100, description="Percentage of personal loan enquiries in last 6 months out of all enquiries ever made", example=25.5)

    num_std_6mts: int = Field(..., ge=0, description="Number of standard accounts in last 6 months", example=3)

    num_std: int = Field(..., ge=0, description="Total number of standard accounts", example=5)

    num_deliq_12mts: int = Field(..., ge=0, description="Number of delinquent accounts in last 12 months", example=1)

    max_recent_level_of_deliq: int = Field(..., ge=0, description="Maximum recent delinquency level", example=2)

    Age_Newest_TL: int = Field(..., ge=0, description="Age of newest tradeline in months", example=12)

    num_times_60p_dpd: int = Field(..., ge=0, description="Number of times payment delayed by more than 60 days", example=0)

    tot_enq: float = Field(..., ge=0, description="Total number of enquiries", example=10)

    PL_enq_L6m: float = Field(..., ge=0, description="Number of personal loan enquiries in last 6 months", example=2)

    time_since_recent_payment_missing: int = Field(..., ge=0, description="Time since most recent missing payment in months", example=8)

    pct_tl_open_L12M: float = Field(..., ge=0, le=100, description="Percentage of tradelines opened in last 12 months", example=35.5)

    Tot_TL_closed_L12M: int = Field(..., ge=0, description="Total tradelines closed in last 12 months", example=2)

    Home_TL: int = Field(..., ge=0, description="Number of home loan tradelines", example=1)

    pct_tl_open_L6M: float = Field(..., ge=0, le=100, description="Percentage of tradelines opened in last 6 months", example=20.0)

    pct_closed_tl: float=Field(..., ge=0, le=100, description="Percent closed accounts", example=20.0)
