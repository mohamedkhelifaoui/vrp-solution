Champion Pack
=============

- champions_manifest.csv : instance → chosen method
- *.json                 : VRPTW solutions (routes) per instance
How to use:
1) Find the row for your instance in champions_manifest.csv to see the chosen method.
2) Load the corresponding JSON (named {instance}__champion_{method}.json).
   Fields:
   - instance : string
   - routes   : list[list[int]]  (customer IDs; depot is implicit 0)
   - feasible : bool

Evaluation settings used to select champions:
- K=200, seed=42, cv_global=0.20, cv_link=0.10
