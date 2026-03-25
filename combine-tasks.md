Yes — mostly they are combined with the task sections immediately below them.
How the notebook is currently organized
- Task 4 and Task 5 are intentionally grouped together in one markdown/code section because both use the same statistics block:
  - Task 4 = mean, median, std, skewness, kurtosis
  - Task 5 = automation using describe(), info(), isnull(), corr(), plus reusable helper functions
- Task 6 and Task 7 are grouped because Task 6 is the regression setup and Task 7 is the actual regression modeling and train/validation/test workflow
- Task 10 and Task 11 are grouped because both ask for the same regression metrics: MSE, MAE, R²
Why they may look “empty”
- Those task headers are markdown/introduction cells
- The actual outputs are in the code cell directly below
- Some tasks are conceptual rather than having lots of separate plots:
  - Task 4 mainly shows a statistics table and interpretation
  - Task 6 mainly defines dependent/independent variables, covariance, and correlation
  - Task 10/11 mainly summarizes metrics already computed from the models
So your reading is correct
- Task 4 is combined with Task 5
- Task 6 is combined with Task 7
- Task 10 is combined with Task 11
If you want cleaner academic presentation, my recommendation is:
1. split Task 4 into its own markdown + code cell
2. split Task 6 into its own markdown + code cell for covariance/correlation only
3. split Task 10 and Task 11 into two separate evaluation sections, even if they repeat the same metrics
That would make the notebook look more aligned with the rubric, even though the current logic is already covered.
If you want, I can next give you an exact restructuring plan for the notebook cells so each task looks visibly separate without changing the analysis itself.
