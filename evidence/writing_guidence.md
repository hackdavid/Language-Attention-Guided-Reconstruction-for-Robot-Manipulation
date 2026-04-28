## Professor guidence for the writng the paper

Programme MSc Artificial Intelligence
Module Title Deep Learning and Generative AI
Module Code CMP030L043
Module Level 7 Assessment Type(s)
Word Length / Duration
P1 2,000 words
P2 3,000 words
% contribution to
module mark
P1 40%
P2 60%
Deadline (date & time) for
Submission
P1
6/3/2026,16:00
P2
17/4/2026,16:00
Format/Location of
submission

Assessment Feedback date:
Learning Outcomes
The primary aims of the module are to:
1. Develop Expertise in Deep Learning
Equip students with an advanced understanding of
deep learning principles, including neural network
architectures, optimisation techniques, and model
regularisation.
2. Enhance Technical Proficiency
Provide hands-on experience with frameworks such as
TensorFlow and PyTorch for designing, training, and
optimising deep learning and generative AI models.
3. Enable Critical Evaluation of Generative AI
Train students to critically assess generative AI models'
strengths, limitations, and ethical implications,
including VAEs, GANs, and diƯusion models.
4. Foster Innovation in AI Applications
Encourage students to design and implement cuttingedge AI solutions that leverage deep learning and
generative models for real-world challenges.
5. Promote Ethical and Scalable AI Development
Ensure students understand the computational
challenges, ethical considerations, and responsible AI
principles involved in deploying deep learning systems
at scale.
Assessment Requirements
This module is assessed via two summative components. To successfully pass the module,
students must achieve a pass mark (50%) in BOTH Component 1 and Component 2.
Component Assessment
Type
Weighting Word
Count
Submission Format Deadline
Part 1 Critical
Appraisal &
Proposal
40% 2,000
words
PDF Document (via
Turnitin)
Friday,6/3/2026
16:00PM
Part 2 Project
(Artefact +
Report)
60% 3,000
words
Report: PDF Document.
Artefact: Link to GitHub
Repository/Colab and
zipped code folder
(Jupyter
Notebooks/Python
scripts).
Friday,17/4/2026
16:00PM
Assessment Details
Part 1: Critical Appraisal & Proposal
You must select one research paper from the provided list or a comparable high-impact paper
published within the last 3 years (subject to tutor approval). Your task is to critically analyse
the paper's methodology and findings and then propose a novel improvement or extension.
Required Structure
1. Summary (approx. 200 words): Briefly summarize the paper’s core contribution,
architecture, and key results.
2. Critical Appraisal (approx. 800 words):
o Evaluate the theoretical foundations (referencing module content like
backpropagation, optimization, or specific architectures).
o Critique the methodology: Are the baselines fair? Is the dataset appropriate?
Are the ablation studies suƯicient?
o Analyse the limitations: What does the model fail to do? What are the
computational costs or ethical risks?
3. Proposal for Improvement (approx. 1000 words):
o Propose a specific, technical extension to the paper (e.g., changing the loss
function, introducing a new attention mechanism, applying the model to a
diƯerent domain, or optimizing inference speed).
o Justify why this improvement is needed based on your critique.
o Hypothesize the expected outcome.
o Note: This proposal will form the basis of your implementation in Part 2.
3. Part 2: Project (Artefact + Report)
Building on the proposal designed in Part 1, you will implement a Deep Learning or Generative
AI solution. You are expected to develop a software artefact using Python (PyTorch or
TensorFlow) and write a report evaluating its performance.
Requirements
1. The Artefact (Code):
o Must be functional, well-commented, and reproducible.
o Must demonstrate the implementation of the improvement/extension
proposed in Part 1.
o Must include a README.md with instructions on how to run the model.
2. The Report (3000 words):
o Methodology: Detail how you implemented the solution. Explain the
architecture, data preprocessing, and training strategy (hyperparameters,
optimization).
o Results & Evaluation: Present quantitative metrics (Accuracy, F1, FID,
Perplexity, etc.) and qualitative analysis (visualizations, sample outputs).
Compare your results against a baseline (e.g., the original paper's
implementation).
o Critical Discussion: Discuss challenges faced, computational constraints,
and why the hypothesis from Part 1 was (or was not) supported.
o Ethics & Scalability: Address the ethical implications (bias, misuse) and
scalability of deploying your model in a real-world setting (referencing LO 5).
Report Format and Guidelines: The report should be approximately 8–10 pages long (A4 size,
using a standard 11 or 12 pt font, single or 1.15 line spacing). This length excludes the cover
page and references (appendices are not expected, but if you include any, keep them brief).
It should be well-structured and written in a clear, formal style. A recommended structure is:
 Title: Project title, your name/student number, module name, and submission date.
 Abstract: Optional. A short paragraph summarising the problem, approach, and key
results.
 Introduction: Introduce the problem you are solving and why it is important. Clearly
state the aims/objectives of your project and the scope. Provide any necessary
background to set the context.
 Background / Literature Review: Briefly review relevant work or methods from
textbooks, research papers, or industry to show your understanding of the domain.
Identify if others have tackled similar problems and what approaches they used. This
section demonstrates awareness of the field.
 Methodology: Describe your approach in detail. This includes the dataset you used
(and how it was collected or sourced), data preprocessing steps, and the design of
your deep learning model. Explain the architecture of your neural network, the
reasoning behind your design choices, and any algorithms or techniques employed
(e.g., transfer learning, regularisation, etc.). If you tried multiple approaches, describe
them.
 Experiments and Results: Explain how you trained the model and present the results.
Include details like training/validation split, training duration or epochs, and any tuning
of hyperparameters. Present results with appropriate metrics (accuracy, F1-score,
MSE, etc. as applicable) and use tables or graphs (learning curves, confusion matrix,
sample outputs) to illustrate performance. Compare results for diƯerent model
versions or against a baseline if available.
 Discussion: Analyse your results. Discuss what the model’s performance indicates
about the problem. Mention any challenges or unexpected outcomes and how you
addressed them. For example, if the model had poor accuracy on certain classes or
the training was unstable, provide insight into why. Relate your findings back to the
objectives – did you meet them? This section is also a place to acknowledge
limitations of your approach and suggest possible improvements or future work.
 Conclusion: Summarise what was accomplished in the project and the key
takeaways. Emphasise the significance of your results and any concluding thoughts
on using deep learning for this application.
 References: List all sources cited in the report (academic papers, online articles,
datasets, libraries, etc.) in a consistent citation format (IEEE). Ensure every external
idea or resource is properly referenced.
Throughout the report, use clear headings and subheadings to guide the reader. Numbered
sections (1. Introduction, 2. Methodology, …) are recommended. Include figure captions and
table titles for any visuals, and refer to them in the text (e.g., “As shown in Figure 1…”). Writing
clarity and organisation are crucial – avoid overly long paragraphs and use formal academic
language.
Code and Repository Guidelines: Your code should be neatly organised and documented.
If using a GitHub repo, include a README.md with instructions on how to run your code or
reproduce your results. If using Colab, ensure that the notebook is well-commented and that
all results (figures, printed metrics) are visible without requiring additional runs (preferably,
have it run top-to-bottom). The markers will not significantly debug your code – it should run
as provided. Using notebooks is fine, but ensure any required data files are accessible (you
may include small data files in the repo or provide a script to download them). Remember to
keep the repository private until submission if it’s not meant to be public (you can invite the
instructor/markers as collaborators or provide a private link) to avoid academic integrity
issues. After grading, you can open-source your project if desired. 


Assessment Success Guidance
To excel in this assessment, follow these key tips:
 Start Early & Plan: Begin immediately. Break the project into manageable phases with
deadlines for research, implementation, analysis, and writing.
 Do Background Research: Before coding, review relevant papers and articles to
inform your approach and strengthen your report.
 Justify Your Choices: Clearly explain your design decisions in your report. Why did
you choose a specific model, architecture, or hyperparameter?
 Experiment Systematically: Go beyond a single run. Perform hyperparameter tuning
or ablation studies to demonstrate a thorough, methodical approach.
 Visualise Your Results: Use plots (e.g., training curves, confusion matrices) and
diagrams to clearly present your findings and model architecture.
 Analyse Results Critically: Don’t just state your results—interpret them. Discuss why
you got them, analyse any errors, acknowledge limitations, and suggest future
improvements.
 Maintain Academic Integrity: Your submitted work must be your own. Use AI tools
responsibly for brainstorming or debugging, but never for writing your report or code.
Disclose any significant AI assistance.
 Present Professionally: Ensure your final report is well-structured, proofread for
errors, and formatted cleanly. A professional presentation makes a strong impression.
 Use Available Resources: Take advantage of all support, including oƯice hours,
forums, and the provided resource links.
Following this guidance can significantly improve your project's quality and your final grade.
Assessment Guidance Support, and Formative Feedback
You are not alone in this process! The teaching team is here to support you through the
assessment.
Here are your avenues for help:
 Assessment Briefing (Week 4): We will hold a dedicated session to detail the project
options, submission requirements, and answer your questions. Attendance is highly
recommended.
 Seminars & Labs: Use lab sessions to get hands-on help from tutors, who will allocate
specific time for project Q&A.
 Moodle Q&A Forum: Post questions on the Moodle forum, which the teaching team
monitors regularly. Please do not share full solutions. Check the forum weekly for
announcements and clarifications.
 Formative Feedback: You are strongly encouraged to submit an optional, one-page
Project Update by Week 7. This is a chance to get early, ungraded feedback on your
progress and direction.
 OƯice Hours: The module leader holds weekly oƯice hours for one-on-one guidance.
You can also request an appointment via the Bookings with me page.
 Use of AI Tools: You may use AI for support, such as brainstorming or coding
assistance, but with these rules:
o DO NOT use AI to write your report or analysis. All final writing must be your
own.
o You are responsible for testing and verifying any AI-generated code.
o Acknowledge any significant AI assistance in your submission.
o Use AI ethically to enhance your learning, not to bypass it. Ask if you are unsure.
 Additional Support: Please inform us early if you require special accommodations or
are facing personal diƯiculties. The university’s Academic Skills Centre is also
available for writing and study support.
 Final Feedback: After grading, you will receive detailed feedback on your work to help
you understand your performance and improve.
We encourage you to be proactive in seeking help. Engaging with these opportunities will
significantly enhance your learning and performance.
Contact for Queries/ who you can contact for further information or queries
You can find information about your tutor on the module’s Moodle page.
Assessment Rubrics
Rubric for Part 1: Critical Appraisal & Proposal (40%)
Criteria Distinction (70-100%) Merit (60-69%) Pass (50-
59%)
Fail (<50%)
Critical
Analysis
(40%)
Exceptional insight.
Deconstructs the paper’s
methodology with precision.
Identifies subtle flaws (e.g.,
data leakage, weak
baselines) and links critique
to broader SOTA literature.
Very good
analysis.
Clearly
identifies
strengths and
weaknesses.
Good
understanding
of the
methodology.
Arguments are
coherent and
well-justified.
Adequate
analysis.
Summarizes
the paper
well but
critique is
generic.
Identifies
obvious
limitations
but lacks
deep
technical
nuance.
Descriptive only.
Fails to critique;
merely repeats
the abstract.
Misunderstands
core concepts.
Proposed
Improvement
(30%)
Novel and feasible. The
proposal demonstrates
technical creativity (e.g., new
loss term, architectural
block) and is
mathematically/theoretically
justified.
Innovative. The
proposal is a
logical
extension of
the critique. It
is technically
sound and
Competent.
Suggests a
standard
extension
(e.g., "add
more
layers")
without deep
Weak/Absent.
Proposal is
unrealistic,
scientifically
invalid, or
missing. No
feasible to
implement.
justification.
Feasibility is
plausible.
logical link to
the critique.
Theoretical
Knowledge
(20%)
Mastery of concepts.
Demonstrates sophisticated
understanding of deep
learning theory (gradients,
manifolds, attention).
Terminology is precise.
Strong
understanding.
Concepts are
applied
correctly. Minor
inaccuracies
do not detract
from the
argument.
Acceptable.
Basic
concepts are
correct, but
there may be
confusion
regarding
complex
topics.
Fundamental
errors.
Significant gaps
in
understanding
deep learning
principles.
Academic
Standards
(10%)
Flawless presentation.
Professional narrative.
Perfect Harvard referencing.
Strict adherence to word
count.
Clear
presentation.
Wellstructured.
Referencing is
largely correct.
Readable
but
disjointed.
Citations
exist but
may have
errors.
Unprofessional.
Poor grammar.
Missing
citations or
plagiarism.
Rubric for Part 2: Project (Artefact + Report) (60%)
Criteria Distinction (70-
100%)
Merit (60-69%) Pass (50-59%) Fail (<50%)
Technical
Artefact
(40%)
Professional quality.
Code is optimized,
modular, and
reproducible.
Implements complex
improvements
successfully.
README is
comprehensive.
High quality. Code
runs correctly and
achieves the goal.
Good use of
libraries
(PyTorch/TF). Minor
style issues.
Functional. Code
works but may
be ineƯicient or
messy. Relies
heavily on
tutorial code
with minimal
modification.
Non-functional.
Code crashes or is
missing.
Plagiarism of
existing
repositories.
Evaluation
& Results
(30%)
Rigorous. Uses
advanced metrics
(FID, Perplexity).
Includes ablation
studies and error
bars. Critical
analysis of why the
model performed as
it did.
Solid. Uses correct
metrics.
Comparison with at
least one baseline.
Interpretation of
results is generally
correct.
Basic. Reports
simple metrics
(accuracy) but
lacks depth. No
rigorous
comparison.
InsuƯicient. No
meaningful
evaluation. Claims
are
unsubstantiated
by data.
Critical
Reflection
(20%)
Insightful. Deep
discussion of
challenges (e.g.,
Good reflection.
Discusses what
worked/didn't.
Descriptive. Lists
steps taken
without
Little/No
reflection. Report
is a diary of tasks
instability) and
solutions. Evaluates
the gap between
theory and practice.
Identifies
limitations of the
approach.
analyzing why.
Mentions
challenges
superficially.
rather than a
critical analysis.
Ethics &
Scalability
(10%)
Sophisticated
handling. Explicitly
addresses specific
ethical risks and
computational
costs. Proposes
mitigations.
Competent
handling. Mentions
ethics/eƯiciency
relevant to the
project. Shows
awareness of
responsible AI.
Basic
awareness.
Generic mention
of ethics but
lacks depth.
Ignores
scalability.
Ignored. No
discussion of
ethics, safety, or
scalability.
Use of Artificial Intelligence (AI)
In this section include details of the use of AI tools to support the assessment.
The assessment is designed so that the use of AI during the assessment is possible.
You must acknowledge any use of AI and appropriately cite all AI generated outputs.
Please make sure you read and understand the assessment guidelines and ask your
Module Leader if you have any questions.
You can find the student guidance on the use of AI in the Library and Nest
Mitigating circumstances/late penalties
Sometimes circumstances outside of your control may affect your studies and might
prevent you from submitting work on time or attending an exam.
The University offers the ability for students to request additional time to complete an
assessment or to defer an examination to a later date. If you are finding yourself in such
a situation, please speak to your Academic Guidance Tutor, the Roehampton Student
Union (RSU) or someone in the Wellbeing team first, who can support you. Further
details can be found on the mitigating circumstances portal.
If you do not apply for or are not approved for Mitigating Circumstances, late penalties
will apply. If work is submitted up to 14 days late, the mark will be capped at 40%/50%
(delete as appropriate); if it is over 14 days late, it will not be marked.
Resubmissions and Reassessment
If you are required to resubmit this assessment or take part in reassessment, you will be
notified via Moodle and your student email. Please ensure you check both regularly. Any
reassessment tasks will follow the same learning outcomes and criteria.
Submission Checklist
Before you submit, ask yourself:
Have I fully answered the assessment brief?
Have I met the word count and formatting requirements?
Is my referencing complete and accurate?
Have I declared any AI use honestly?
Have I proofread my work?
Am I submitting through the correct platform before the deadline?