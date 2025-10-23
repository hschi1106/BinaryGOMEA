# BinaryGOMEA

1. ```make```
2. ```./GOMEA -h``` prints out the full information about GOMEA usage

Usage: GOMEA [-h] ...
   -h: Prints out this usage information.

General settings:
    --L: Number of variables. Default: 1
    --alphabet: Alphabet size. Default: 2 (binary optimization)
    --problem: Index of optimization problem to be solved (maximization). Default: 0
    --instance: Problem instance.
    --vtr: Value To Reach value.
    --time: Time Limit (in seconds).
    --folder: Folder where to save results.
    --seed: Random seed.
    -partial: Enables partial evaluations. Default: disabled.
    -saveEvals: Enables saving all evaluations in archive. Default: disabled

GOMEA Configuration settings:
    --GOM: GOM type. Default: 0 (LT). 1 - conditionalGOM (CGOM) based on MI
    --threshold: Threshold value for CGOM. Default: 0.8
    --hillClimber: Hill Climber usage. Default: 0 (no HC). 1 - single HC. 2 - Exhaustive HC
    --FOS: FOS type. Default: 1 (LT). 2 - Filtered LT. 3 - Efficient implementation of LT for P3 (without Tournament Selection). 4 - Efficient implementation of filtered LT for P3 (without Tournament Selection).
    --orderFOS: FOS order. Default: 0 (randomly shuffled). 1 - sorted by the FOS elements size (ascending order).
    --similarityMeasure: FOS building similarity measure. Default: 0 (MI). 1 - NMI.
    --scheme: Population Management Scheme. Default: 0 (single population). 1 - IMS. 2 - P3. 3 - P3-MI (Quadratic). 4 - P3-MI (Linear).
    --populationSize: Population Size for single population run.
    -FI: Enables Forced Improvements. Default: disabled.
    -donorSearch: Enables Exhaustive Donor Search Default: disabled.
    -tournamentSelection: Enables Tournament Selection of size 2 prior to Linkage Model Learning

指令:
./GOMEA --L 40 --problem 1_5_4 --populationSize 200 --time 10 --FOS 1 --vtr 50

--L 長度
--problem 問題編號_k_steps
eg: cyclictrap 重疊一個bit
1_5_4 
--populationSize 
--time 設定時間 可以長一點
--FOS 1 
--vtr maxfitness(你也可以寫在problems.cpp)

跑完指令會生成檔案結果在
./dumps/elitists.txt

加問題：./src/problems.cpp+./include/problems.hpp

1.createProblemInstance+problemNameByIndex 多加一個case
2.然後在.cpp裡面新增問題的class，架構如附圖.h檔(包含initializeProblem、calculateFitness)，可以參考gomea自己寫的!

./GOMEA --L 40 --problem 7_6_5 --populationSize 200 --time 100 --FOS 1 --vtr 50
