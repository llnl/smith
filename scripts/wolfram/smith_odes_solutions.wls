#!/usr/bin/env wolframscript

(* 
Copyright (c) Lawrence Livermore National Security, LLC and
other Smith Project Developers. See the top-level LICENSE file for
details.

SPDX-License-Identifier: (BSD-3-Clause)
*)

M := ( {
    {2, 1, 0},
    {1, 4, 1},
    {0, 1, 2}
   } );
CC := ( {
    {1, 0, 0},
    {0, 1, 0},
    {0, 0, 1}
   } );
KL := ( {
    {1, -1, 0},
    {-1, 2, -1},
    {0, -1, 1}
   } );
KNL[t_] := ( {
    {1, -1, 0},
    {-1, 2 - x2[t], -1 + x2[t]},
    {0, -1 + x3[t], 1 + 0.5 x3[t]}
   } );

f := {1, 2, 3};


FirstOrderLinearTest[X1_ : "default"] := Module[{x, x0, which, soln},
  If[X1 === "default",
   x[t_] := {x1[t], x2[t], x3[t]};
   x0 = {1, 2, 3};
   which = {1, 2, 3};
   ,
   x[t_] := {X1[t], x2[t], x3[t]};
   x0 = {X1[0], 2, 3};
   which = {2, 3};
   ];
  
  soln = First[NDSolve[{
      (M.x'[t] + KL.x[t])[[which]] == f[[which]],
      x[0] == x0
      },
     x[t][[which]],
     {t, 0, 1}
     ]];
  
  Print["Exact Solution: ", x[t] /. soln /. t -> 1.0];
]

FirstOrderNonlinearTest[X1_ : "default"] := 
 Module[{x, x0, which, soln},
  If[X1 === "default",
   x[t_] := {x1[t], x2[t], x3[t]};
   x0 = {1, 2, 3};
   which = {1, 2, 3};
   ,
   x[t_] := {X1[t], x2[t], x3[t]};
   x0 = {X1[0], 2, 3};
   which = {2, 3};
   ];
  
  soln = First[NDSolve[{
      (M.x'[t] + KNL[t].x[t])[[which]] == f[[which]],
      x[0] == x0
      },
     x[t][[which]],
     {t, 0, 1}
     ]];
  
  Print["Exact Solution: ", x[t] /. soln /. t -> 1.0];
]

SecondOrderLinearTest[X1_ : "default"] := 
  Module[{x, x0, v0, which, soln},
   If[X1 === "default",
    x[t_] := {x1[t], x2[t], x3[t]};
    x0 = {1, 2, 3};
    v0 = {1, -1, 0};
    which = {1, 2, 3};
    ,
    x[t_] := {X1[t], x2[t], x3[t]};
    x0 = {X1[0], 2, 3};
    v0 = {X1'[0], -1, 0};
    which = {2, 3};
    ];
   
   soln = First[NDSolve[{
       (M.x''[t] + CC.x'[t] + KL.x[t])[[which]]  == f[[which]],
       x[0][[which]] == x0[[which]],
       x'[0][[which]] == v0[[which]]
       },
      x[t][[which]],
      {t, 0, 1}
      ]];
   
   Print["Exact Solution: ", x[t] /. soln /. t -> 1.0];
   Print["Exact Derivative: ", D[x[t] /. soln, t] /. t -> 1.0];
];

SecondOrderNonlinearTest[X1_ : "default"] := 
  Module[{x, x0, v0, which, soln},
   If[X1 === "default",
    x[t_] := {x1[t], x2[t], x3[t]};
    x0 = {1, 2, 3};
    v0 = {1, -1, 0};
    which = {1, 2, 3};
    ,
    x[t_] := {X1[t], x2[t], x3[t]};
    x0 = {X1[0], 2, 3};
    v0 = {X1'[0], -1, 0};
    which = {2, 3};
    ];
   
   soln = First[NDSolve[{
       (M.x''[t] + CC.x'[t] + KNL[t].x[t])[[which]]  == f[[which]],
       x[0][[which]] == x0[[which]],
       x'[0][[which]] == v0[[which]]
       },
      x[t][[which]],
      {t, 0, 1}
      ]];
   
   Print["Exact Solution: ", x[t] /. soln /. t -> 1.0];
   Print["Exact Derivative: ", D[x[t] /. soln, t] /. t -> 1.0];
];


Print["Unconstrained, Linear, First-Order:"]
FirstOrderLinearTest[]

Print["Sine Wave, Linear, First-Order:"]
FirstOrderLinearTest[1 + Sin[2 \[Pi] #] &]

Print["Triangle Wave, Linear, First-Order:"]
FirstOrderLinearTest[1 + TriangleWave[#] &]

Print["Unconstrained, Nonlinear, First-Order:"]
FirstOrderNonlinearTest[]

Print["Sine Wave, Nonlinear, First-Order:"]
FirstOrderNonlinearTest[1 + Sin[2 \[Pi] #] &]

Print["Triangle Wave, Nonlinear, First-Order:"]
FirstOrderNonlinearTest[1 + TriangleWave[#] &]

Print["Unconstrained, Linear, Second-Order:"]
SecondOrderLinearTest[]

Print["Sine Wave, Linear, Second-Order:"]
SecondOrderLinearTest[1 + Sin[2 \[Pi] #] &]

Print["Triangle Wave, Linear, Second-Order:"]
SecondOrderLinearTest[1 + TriangleWave[#] &]

Print["Unconstrained, Nonlinear, Second-Order:"]
SecondOrderNonlinearTest[]

Print["Sine Wave, Nonlinear, Second-Order:"]
SecondOrderNonlinearTest[1 + Sin[2 \[Pi] #] &]

Print["Triangle Wave, Nonlinear, Second-Order:"]
SecondOrderNonlinearTest[1 + TriangleWave[#] &]

