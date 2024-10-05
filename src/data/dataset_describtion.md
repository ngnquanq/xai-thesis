#  Overview

This dataset captures customer activity and service usage for a telecommunications company. It includes various metrics related to customer behavior, service plans, usage patterns, and customer interactions. The data can be used to analyze customer behavior and predict churn rates.

# Data Dictionary
| Column Name | Data Type | Description |
| --- | --- | --- |
| State | object | The U.S. state where the customer resides (e.g., KS for Kansas, OH for Ohio). |
| Account length | int64 | The duration (in days or months) that the customer's account has been active. |
| Area code | int64 | The telephone area code associated with the customer's phone number. |
| International plan | object | Indicates whether the customer is subscribed to an international calling plan (Yes or No). |
| Voice mail plan | object | Indicates whether the customer has a voice mail plan (Yes or No). |
| Number vmail messages | int64 | The number of voice mail messages the customer has received. |
| Total day minutes | float64 | The total number of calling minutes the customer used during the daytime. |
| Total day calls | int64 | The total number of calls made by the customer during the daytime. |
| Total day charge | float64 | The total charges incurred by the customer for daytime calls. |
| Total eve minutes | float64 | The total number of calling minutes used during the evening. |
| Total eve calls | int64 | The total number of calls made during the evening. |
| Total eve charge | float64 | The total charges incurred for evening calls. |
| Total night minutes | float64 | The total number of calling minutes used during the night. |
| Total night calls | int64 | The total number of calls made during the night. |
| Total night charge | float64 | The total charges incurred for night calls. |
| Total intl minutes | float64 | The total number of international calling minutes used. |
| Total intl calls | int64 | The total number of international calls made. |
| Total intl charge | float64 | The total charges incurred for international calls. |
| Customer service calls | int64 | The number of calls the customer has made to customer service. |
| Churn | bool | Indicates whether the customer has discontinued the service (True for churned, False for active customers). |