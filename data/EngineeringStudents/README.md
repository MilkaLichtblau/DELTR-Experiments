# The Engineering Students Dataset

The Engineering Students dataset corresponds to the task of sorting a list of applicants to a Chilean 
school of engineering by predicted academic performance. It contains information from first-year students
over 5 years with on average 675 students per year. 

For each student, the following features are available: 
1. their average high-school grades
2. their scores in math, language, and science in a standardized test named PSU test; 
3. the number of university credits taken, passed, and failed during the first year; and their average 
grades at the end of their first year. This information is used to calculate the relevance score.
4. their gender
5. whether they come from a public high school or from a private one

Due to privacy reasons, we cannot upload the original dataset and this folder provides only a pre-processed version.
We created a five-fold cross-validation setup in which each fold contains four years for training 
and one year for testing. 
The judgment score is created by sorting students by decreasing grades upon finishing the first year, 
in which grades are weighted by the credits of each course they passed and divided by the total number 
of credits they took.

The column headers are as follows: `['query_id', 'protected_attribute', 'psu_math', 'psu_language', 'psu_science', 'high_school_grades', 'relevance_score']`

### Working With the Original Dataset
In order to obtain the original dataset, please send a request to [chato@chato.cl](chato@chato.cl) with subject line "Engineering dataset" to request the dataset for research-only purposes.