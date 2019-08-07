\usepackage[dvipsnames]{xcolor,colortbl}
\usepackage{forloop}
\newcounter{loopcntr}
\newcommand{\rpt}[2][1]{%
  \forloop{loopcntr}{0}{\value{loopcntr}<#1}{#2}%
}
\newcommand{\on}[1][1]{
  \forloop{loopcntr}{0}{\value{loopcntr}<#1}{&\cellcolor{ggray}}
}
\newcommand{\ong}[1][1]{
  \forloop{loopcntr}{0}{\value{loopcntr}<#1}{&\cellcolor{Green}}
}
\newcommand{\onr}[1][1]{
  \forloop{loopcntr}{0}{\value{loopcntr}<#1}{&\cellcolor{Red}}
}
\newcommand{\ony}[1][1]{
  \forloop{loopcntr}{0}{\value{loopcntr}<#1}{&\cellcolor{Yellow}}
}
\newcommand{\off}[1][1]{
  \forloop{loopcntr}{0}{\value{loopcntr}<#1}{&}
}

\definecolor{orange}{HTML}{FF7F00}
\definecolor{ggray}{HTML}{bbbbbb}


\begin{landscape}

\noindent\begin{tabular}{|p{0.22\textwidth}*{39}{|p{0.002\textwidth}}|}
\hline
\textbf{} & \multicolumn{3}{c!{\vrule width 0.4mm}}{2018} 
          & \multicolumn{12}{c!{\vrule width 0.4mm}}{2019}
          & \multicolumn{12}{c!{\vrule width 0.4mm}}{2020}
          & \multicolumn{12}{c!{\vrule width 0.4mm}}{2021}\\
           
\hline

\textbf{} & \multicolumn{3}{c!{\vrule width 0.1mm}}{\tiny{Oct-Dec}}
		  & \multicolumn{3}{c!{\vrule width 0.1mm}}{\tiny{Jan-Mar}}
		  & \multicolumn{3}{c!{\vrule width 0.1mm}}{\tiny{Apr-Jun}}
		  & \multicolumn{3}{c!{\vrule width 0.1mm}}{\tiny{Jul-Sep}}
		  & \multicolumn{3}{c!{\vrule width 0.1mm}}{\tiny{Oct-Dec}}
		  & \multicolumn{3}{c!{\vrule width 0.4mm}}{\tiny{Jan-Mar}}
		  & \multicolumn{3}{c!{\vrule width 0.4mm}}{\tiny{Apr-Jun}}
		  & \multicolumn{3}{c!{\vrule width 0.4mm}}{\tiny{Jul-Sep}}
		  & \multicolumn{3}{c!{\vrule width 0.4mm}}{\tiny{Oct-Dec}}
		  & \multicolumn{3}{c!{\vrule width 0.4mm}}{\tiny{Jan-Mar}}
		  & \multicolumn{3}{c!{\vrule width 0.4mm}}{\tiny{Apr-Jun}}
		  & \multicolumn{3}{c!{\vrule width 0.4mm}}{\tiny{Jul-Sep}}
		  & \multicolumn{3}{c!{\vrule width 0.4mm}}{\tiny{Oct-Dec}}\\
           
\hline

\footnotesize{Year 1} \off[1] \on[12] \off[26] \\ \hline
\footnotesize{Literature review} \off[1] \ong[4] \off[34] \\ \hline
\footnotesize{Research proposal} \off[1] \ong[5] \onr[1] \off[32] \\ \hline
\footnotesize{Survival prediction baseline} \off[2] \ong[3] \off[34] \\ \hline
\footnotesize{Spatial Transformer Networks} \off[4] \ong[2] \ony[3] \off[30] \\ \hline
\footnotesize{Survival analysis techniques} \off[9] \ony[4] \off[26] \\ \hline
\footnotesize{Major review} \off[12] \onr[1] \off[26] \\ \hline
\footnotesize{Year 2} \off[13] \on[12] \off[14] \\ \hline
\footnotesize{CVPR deadline} \off[13] \onr[1] \off[25] \\ \hline
\footnotesize{Model interpretation techniques} \off[13] \ony[5] \off[21] \\ \hline
\footnotesize{MICCAI deadline} \off[17] \onr[1] \off[21] \\ \hline
\footnotesize{Reinforcement learning} \off[18] \ony[7] \off[14] \\ \hline
\footnotesize{Major review} \off[24] \onr[1] \off[14] \\ \hline
\footnotesize{Year 3} \off[25] \on[12] \off[2] \\ \hline
\footnotesize{CVPR deadline} \off[25] \onr[1] \off[13] \\ \hline
\footnotesize{Longitudinal aware methods} \off[25] \ony[8] \off[6] \\ \hline
\footnotesize{MICCAI deadline} \off[29] \onr[1] \off[9] \\ \hline
\footnotesize{Thesis}  \off[33] \ony[3] \onr[1] \off[2] \\ \hline

\end{tabular}