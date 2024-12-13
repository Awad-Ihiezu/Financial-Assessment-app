import streamlit as st
import pandas as pd
import Joblib
import time

# Streamlit app
st.title("💵Financial Assessment App")
st.info("The app helps to provide assessments of financial status for a potential loan approval")

# Load the model
try:
    model1 = joblib.load('risk_score_model.pkl')
    model2 = joblib.load('loan_approval_model.pkl')
    model3 = joblib.load('loan_amount_model.pkl')
except Exception as e:
    st.error(f"Failed to load models: {e}")

with st.expander("Data"):
    st.write('**Raw Data**')
    df = pd.read_csv('https://raw.githubusercontent.com/Awad-Ihiezu/Financial-Assessment-app/refs/heads/main/Loan.csv')
    out = ["ApplicationDate", "Experience", "NumberOfDependents", "LoanPurpose", "JobTenure"]
    df.drop(out, axis= 1, inplace= True)
    df
    
    st.write('**Cleaned Data**')
    df2 = pd.read_csv('https://raw.githubusercontent.com/Awad-Ihiezu/Financial-Assessment-app/refs/heads/main/cleaned.csv')
    out = ["Experience", "NumberOfDependents", "LoanPurpose", "JobTenure"]
    df2.drop(out, axis= 1, inplace= True)
    df2

# Input Fields for Models
with st.sidebar:
    st.header("Input Features")
    Age = st.slider("Age", 18, 82, 45)
    AnnualIncome = st.number_input("Annual Income", value=0)
    CreditScore = st.slider("Credit Score", 340, 720, 500)
    EmploymentStatus = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"])
    EducationLevel = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "Associate", "Doctorate"])
    LoanAmount = st.number_input("Loan Amount", value=0)
    LoanDuration = st.slider("Loan Duration", 12, 120, 60)
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    HomeOwnershipStatus = st.selectbox("Home Ownership Status", ["Own", "Mortgage", "Rent", "Other"])
    MonthlyDebtPayments = st.number_input("Monthly Debt Payments", value=0)
    CreditCardUtilizationRate = st.number_input("Credit Card Utilization Rate")
    NumberOfOpenCreditLines = st.slider("Number Of Open CreditLines", 0, 14, 7)
    NumberOfCreditInquiries = st.slider("Number Of Open Inquiries", 0, 8, 4)
    DebtToIncomeRatio = st.number_input("Debt To Income Ratio")
    BankruptcyHistory = st.selectbox("Bankruptcy History", ["Yes", "No"])
    PreviousLoanDefaults = st.selectbox("Previous Loan Defaults", ["Yes", "No"])
    PaymentHistory = st.slider("Payment History", 1, 45, 20)
    LengthOfCreditHistory = st.slider("Length Of Credit History", 1, 30, 15)
    SavingsAccountBalance = st.number_input("Savings Account Balance", value=0)
    CheckingAccountBalance = st.number_input("Checking Account Balance", value=0)
    TotalAssets = st.number_input("Total Assets", value=0)
    TotalLiabilities = st.number_input("Total Liabilities", value=0)
    MonthlyIncome = st.number_input("Monthly Income")
    UtilityBillsPaymentHistory = st.number_input("Utility Bills Payment History")
    NetWorth = st.number_input("NetWorth", value=0)
    BaseInterestRate = st.number_input("Base Interest Rate")
    InterestRate = st.number_input("Interest Rate")
    MonthlyLoanPayment = st.number_input("Monthly Loan Payment")
    TotalDebtToIncomeRatio = st.number_input("Total Debt To Income Ratio")
# Add more inputs as needed

# Create a dataframe for input features
data = {
    'Age': Age, 'AnnualIncome': AnnualIncome, 'CreditScore': CreditScore, 'EmploymentStatus': EmploymentStatus,
    'EducationLevel': EducationLevel, 'LoanAmount': LoanAmount, 'LoanDuration': LoanDuration,
    'MaritalStatus': MaritalStatus, 'HomeOwnershipStatus': HomeOwnershipStatus, 'MonthlyDebtPayments': MonthlyDebtPayments,
    'CreditCardUtilizationRate': CreditCardUtilizationRate, 'NumberOfOpenCreditLines': NumberOfOpenCreditLines,
    'NumberOfCreditInquiries': NumberOfCreditInquiries, 'DebtToIncomeRatio': DebtToIncomeRatio,
    'BankruptcyHistory': BankruptcyHistory, 'PreviousLoanDefaults': PreviousLoanDefaults, 'PaymentHistory': PaymentHistory,
    'LengthOfCreditHistory': LengthOfCreditHistory, 'SavingsAccountBalance': SavingsAccountBalance, 'CheckingAccountBalance': CheckingAccountBalance,
    'TotalAssets': TotalAssets, 'TotalLiabilities': TotalLiabilities, 'MonthlyIncome': MonthlyIncome,
    'UtilityBillsPaymentHistory': UtilityBillsPaymentHistory, 'NetWorth': NetWorth, 'BaseInterestRate': BaseInterestRate,
    'InterestRate': InterestRate, 'MonthlyLoanPayment': MonthlyLoanPayment, 'TotalDebtToIncomeRatio': TotalDebtToIncomeRatio
}
input_df = pd.DataFrame(data, index=[0])
with st.expander("Test Sample"):
    st.write("**44, 105639, 627, 0, 0, 20409, 108, 3, 2, 387, 0.282174466, 1, 4, 0.327078397, 0, 0, 22, 21, 3379, 5476, 165235, 22300, 8803.25, 0.623227567, 142935, 0.251909, 0.263740227, 495.9515749, 0.100298364**")
st.write("**Input Features**")
input_df


# Change the dropdown selections to numerical values
employment_status_mapping = {"Employed": 0, "Self-Employed": 1, "Unemployed": 2}
EmploymentStatus = employment_status_mapping[EmploymentStatus]

educationLevel_mapping = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'Associate': 3, 'Doctorate': 4}
EducationLevel = educationLevel_mapping[EducationLevel]

maritalStatus_mapping = {"Single": 0, "Married": 1, "Divorced": 2, "Widowed": 3}
MaritalStatus = maritalStatus_mapping[MaritalStatus]

homeOwnershipStatus_mapping = {'Own': 0, 'Mortgage': 1, 'Rent': 2, 'Other': 3}
HomeOwnershipStatus = homeOwnershipStatus_mapping[HomeOwnershipStatus]

bankruptcyHistory_mapping = {"No": 0, "Yes": 1}
BankruptcyHistory = bankruptcyHistory_mapping[BankruptcyHistory]

previousLoanDefaults_mapping = {"No": 0, "Yes": 1}
PreviousLoanDefaults = previousLoanDefaults_mapping[PreviousLoanDefaults]

# Model 1: Risk Score Prediction
if st.button("Predict Risk Score"):
    features1 = [Age, AnnualIncome, CreditScore, EmploymentStatus, EducationLevel, LoanAmount, LoanDuration, MaritalStatus, 
        HomeOwnershipStatus, MonthlyDebtPayments, CreditCardUtilizationRate, NumberOfOpenCreditLines, NumberOfCreditInquiries, 
        DebtToIncomeRatio, BankruptcyHistory, PreviousLoanDefaults, PaymentHistory, LengthOfCreditHistory, SavingsAccountBalance, 
        CheckingAccountBalance, TotalAssets, TotalLiabilities, MonthlyIncome, UtilityBillsPaymentHistory, NetWorth, 
        BaseInterestRate, InterestRate, MonthlyLoanPayment, TotalDebtToIncomeRatio]
    
    # Progress bar
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.03)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()

    prediction1 = model1.predict([features1])
    st.write(f"Risk Score:  {prediction1[0]}")

# Model 2: Loan Approval Prediction
if st.button("Predict Loan Approval"):
    features2 = [Age, AnnualIncome, CreditScore, EmploymentStatus, EducationLevel, LoanAmount, LoanDuration, MaritalStatus, 
        HomeOwnershipStatus, MonthlyDebtPayments, CreditCardUtilizationRate, NumberOfOpenCreditLines, NumberOfCreditInquiries, 
        DebtToIncomeRatio, BankruptcyHistory, PreviousLoanDefaults, PaymentHistory, LengthOfCreditHistory, SavingsAccountBalance, 
        CheckingAccountBalance, TotalAssets, TotalLiabilities, MonthlyIncome, UtilityBillsPaymentHistory, NetWorth, 
        BaseInterestRate, InterestRate, MonthlyLoanPayment, TotalDebtToIncomeRatio]

    # Progress bar
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.03)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()

    prediction2 = model2.predict([features2])
    if prediction2 == 1:
        prediction2 = "Loan Approved"
        st.write(f"Predicted:  {prediction2}")
    else:
        prediction2 = "Loan Disapproved"
        features3 = [CreditScore, LoanDuration, MonthlyLoanPayment, InterestRate, BaseInterestRate,
                    TotalDebtToIncomeRatio, NetWorth, LengthOfCreditHistory, DebtToIncomeRatio, TotalLiabilities]
        potential_loan_amount = model3.predict([features3])[0]
        st.write(f"Predicted:  {prediction2}")
        st.write(f"Potential Loan Amount that could be approved: ${potential_loan_amount:,.2f}")
