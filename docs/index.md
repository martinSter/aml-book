--- 
title: "Machine Learning für das KMU"
author: "Martin Sterchi"
date: "2024-03-22"
site: bookdown::bookdown_site
documentclass: book
bibliography: [book.bib, packages.bib]
# url: your book url like https://bookdown.org/yihui/bookdown
# cover-image: path to the social sharing image like images/cover.jpg
description: |
  This is a minimal example of using the bookdown package to write a book.
  The HTML output format for this example is bookdown::bs4_book,
  set in the _output.yml file.
biblio-style: apalike
csl: chicago-fullnote-bibliography.csl
---

# Über das Buch {-}

Die Motivation für dieses Buch kam aus der Erkenntnis, dass viele kleine und mittelgrosse Unternehmen (KMU) in der Schweiz zwar über grosse Datenmengen verfügen, aber nicht das nötige Knowhow haben, um die Daten zu analysieren und für die Optimierung von Entscheidungsprozessen zu nutzen. Mit diesem Buch möchte ich einen kleinen Beitrag leisten, den Knowhow Transfer von Fachhochschulen in die Unternehmen zu katalysieren.

Das Buch versucht, sowohl die klassischen Machine Learning Methoden als auch neueste Entwicklungen im Deep Learning mit einem Fokus auf die Anwendung zu vermitteln. Deep Learning kann als eine Teilmenge des Machine Learnings gesehen werden. Das heisst, jede Deep Learning Methode ist automatisch auch eine Machine Learning Methode. Machine Learning entält jedoch weitere Methoden, welche nicht dem Deep Learning zugeordnet werden können. Das Gebiet Machine Learning ist wiederum eine Teilmenge der Methoden der Künstlichen Intelligenz. Letztere enthält weitere Methoden, welche nicht dem Machine Learning zuzuordnen sind. Abbildung \@ref(fig:kimldl) versucht diesen Sachverhalt schematisch darzustellen.

<div class="figure" style="text-align: center">
<img src="images/KI_ML_DL.png" alt="Unterscheidung zwischen KI, ML und DL. " width="60%" />
<p class="caption">(\#fig:kimldl)Unterscheidung zwischen KI, ML und DL. </p>
</div>

Wir werden im ganzen Buch die folgenden (üblichen) Abkürzungen verwenden:

* Künstliche Intelligenz = KI (oft spricht man auch von AI, was die Abkürzung für den englischen Begriff *Artificial Intelligence* ist).
* Machine Learning = ML
* Deep Learning = DL

Obwohl das Buch einen anwendungsorientierten Ansatz verfolgt, soll die mathematisch-statistische Intuition hinter den beschriebenen Modellen und Methoden nicht zu kurz kommen. Diese Intuition ist aus meiner Sicht zwingend, um beurteilen zu können, ob sich ein Modell überhaupt für ein gegebenes Problem eignet. Am Schluss geht es nämlich darum, dass wir mit dem Einsatz von Machine Learning einen Mehrwert für ein Unternehmen oder für die Gesellschaft schaffen können. Das erfordert, dass wir uns eingehend und kritisch mit den Modellen und deren Eignung für ein gegebenes Problem auseinander setzen.

## Zielgruppe {-}

Das Buch richtet sich insbesondere an Fachhochschulstudierende in der deutschsprachigen Schweiz mit einem intrinsischen Interesse an quantitativen Methoden im Allgemeinen und Machine Learning im Besonderen. Vorausgesetzt werden Mathematikkenntnisse auf Stufe Mittelschule (Berufs- oder gymnasiale Matur), d.h. Sie sollten vertraut sein mit den Grundlagen bezüglich mathematischer Funktionen, der Integral- und Differentialrechnung sowie den wichtigsten Resultaten aus der Algebra. Ausserdem gehe ich davon aus, dass Sie bereits eine Einführung in das Thema Statistik besucht haben und Konzepte aus der deskriptiven Statistik (Mittelwert, Median, Varianz, Quantile, etc.) sowie aus der Inferenzstatistik (Verteilungen, statistisches Testen, etc.) bekannt sind.

Bevor Sie sich aber nun Sorgen machen: Kapitel \@ref(basics) enthält eine Einführung in die wichtigsten Mathematik- und Statistikgrundlagen, die nötig sind für das Verständnis von Machine Learning Modellen.

Da ich mit diesem Buch einen anwendungsorientierten Ansatz verfolge, werden wir auch in das Programmieren einsteigen. Dazu verwenden wir in diesem Buch die Programmiersprache `R`. Es werden keine Vorkenntnisse vorausgesetzt. Kapitel \@ref(intro-R) enthält eine kurze Einführung in die Programmiersprache `R` und verweist Sie auf weiterführende Ressourcen zum Thema Programmieren. Jedes Modell, das wir uns anschauen werden, ist mit R-Code dokumentiert, so dass Sie lernen, wie die Modelle in der Praxis angewendet werden können.

## Aufbau des Buchs {-}

Das Buch enthält folgende Kapitel:

* Kapitel \@ref(intro): Einführung in das Thema Machine Learning mit **Definitionen** sowie Anwendungsbeispielen.
* Kapitel \@ref(basics): Wichtigste **Mathematik- und Statistikgrundlagen**, die für das Verständnis der Modelle in den späteren Kapitel elementar sind.
* Kapitel \@ref(intro-R): Einführung in das **Programmieren** mit `R` sowie Überblick über die wichtigsten `R`-Packages, die wir verwenden werden.
* Kapitel \@ref(lin-reg): Hier erlernen wir die Grundmodelle, um **Regressionsprobleme** zu lösen. Es sind lineare Modelle, was bedeutet, dass die funktionale Form der Modelle linear von den Parametern des Modells abhängen. Grafisch bedeutet dies, dass ein solches Modell im einfachsten Fall durch eine Gerade beschrieben werden kann. 
* Kapitel \@ref(lin-class): In diesem Kapitel lernen wir die Grundmodelle für das **Klassifikationsproblem** kennen. Diese Modelle führen typischerweise zu einer linearen Entscheidungsgrenze (engl. *Decision Boundary*) zwischen den verschiedenen Klassen, die wir unterscheiden oder klassifizieren wollen.
* Kapitel \@ref(ml-pipeline): Damit wir ML in der Praxis anwenden können, lernen wir hier die typische **ML-Pipeline** kennen. Sie werden die Techniken und Methoden kennen lernen, die es braucht, um überhaupt erst an den Punkt zu kommen, um ein ML-Modell rechnen zu können. Oft werden diese Techniken und Methoden unter dem Begriff Preprocessing der Daten zusammengefasst. Doch die Pipeline endet nicht mit dem Rechnen eines ML-Modells. Danach muss ein Modell evaluiert werden und wenn Sie als Analyst\*in zufrieden sind, müssen Sie sich Gedanken machen, wie das Deployment des Modells aussehen soll. Das heisst, wie kann Ihr Modell Dritten zur Verfügung gestellt werden? Wir werden uns hier auch kurz mit den wichtigsten Techniken aus dem Unsupervised Learning befassen.
* Kapitel \@ref(trees): Nach den ersten linearen Modellen für das Regressions- und Klassifikationsproblem lernen wir hier ein flexibleres Modell kennen, nämlich den **Entscheidungsbaum** (engl. *Decision Tree*). Entscheidungsbäume eignen sich sowohl für das Regressions- als auch für das Klassifikationsproblem. Obwohl sie in realen Projekten typischerweise anderen Modellen unterlegen sind, wenn es um die Vorhersagequalität geht, sind sie trotzdem attraktive Modelle, da sie gut visualisierbar sind.
* Kapitel \@ref(ensembles): Aufbauend auf den Entscheidungsbäumen aus dem vorherigen Kapitel können sehr mächtige Modelle erstellt werden, die in der Praxis oft die besten Vorhersagen liefern. Weil es sich dabei üblicherweise um eine clevere Aggregierung der Resultate einer grossen Anzahl individueller Entscheidungsbäume handelt, werden diese Modelle **Ensembles** genannt. Wie die individuellen Entscheidungsbäume eignen sich Ensembles sowohl für das Regressions- als auch für das Klassifikationsproblem.
* Kapitel \@ref(svm): Ein weiteres mächtiges Modell, das sich sowohl für das Regressions- als auch für das Klassifikationsproblem eignet, sind die **Support Vector Machines**. Ihre Popularität ist mit dem Aufstieg von Deep Learning etwas verblasst. Es lohnt sich aber immer noch allemal, diese Familie von Modellen kennen zu lernen, insbesondere auch weil sie nicht als Blackbox-Modelle gelten und theoretisch gut fundiert sind.
* Kapitel \@ref(ann): Ab diesem Kapitel steigen wir in das Thema Deep Learning ein. Sie werden die Architektur von einfachen **Articial Neural Networks** (ANNs) kennen lernen. Ausserdem schauen wir uns in diesem Kapitel den genialen Backpropagation Algorithmus anhand eines einfachen linearen Regressionsproblems an. Dieser Algorithmus ist der Schlüssel für die viel diskutierten Fortschritte im Bereich der künstlichen Intelligenz, weil er das Trainieren von riesigen Modellen überhaupt erst möglich macht.
* Kapitel \@ref(cnn): Hier lernen wir sogenannte **Convolutional Neural Networks** (CNNs) kennen. Sie sind die Basis für die Fortschritte auf dem Gebiet Computer Vision und erlauben beispielsweise Anwendungen im Bereich automatische Gesichtserkennung in Bildern oder Videos.
* Kapitel \@ref(rnn): Nach ANNs und CNNs lernen wir hier **Recurrent Neural Networks** (RNNs) kennen. Diese Modelle bilden die Basis für Probleme, in denen die Daten als Sequenzen vorliegen. Das können einache Zeitreihen (z.B. Börsenkurse) sein, aber auch komplexere Sequenzdaten wie beispielsweise geschriebene oder gesprochene Sprache oder Tonaufnahmen.
* Kapitel \@ref(gen-AI): In diesem letzten Kapitel geht es schliesslich um **Generative KI**. Wir beschäftigen uns hier also mit Modellen, die nicht nur einfach ein Vorhersageprobleme lösen können, sondern auch neue Inhalte (z.B. Texte, Musik, Bilder) generieren können. Abbildung \@ref(fig:genAIExML) enthält als Beispiel den Output einer generativen Software, die basierend auf einem Prompt ein Bild erstellt. Nach dem Lesen dieses Kapitels sollten Sie ein grundlegendes Verständnis für die Funktionsweise von Modellen wie Chat-GPT haben.

<div class="figure" style="text-align: center">
<img src="images/title_picture_ML.png" alt="Beispielsoutput einer generativen Bildgenerierungssoftware (https://stablediffusionweb.com/) basierend auf dem Prompt &quot;A title image for a textbook about Machine Learning targeting small and medium companies.&quot;" width="70%" />
<p class="caption">(\#fig:genAIExML)Beispielsoutput einer generativen Bildgenerierungssoftware (https://stablediffusionweb.com/) basierend auf dem Prompt "A title image for a textbook about Machine Learning targeting small and medium companies."</p>
</div>

## Weiterführende Literatur {-}

Ein grosser Teil des vorliegenden Buchs baut auf bestehenden Büchern zum Thema Machine Learning auf. Ich werde im Buch immer wieder auf die Quellen verweisen. Die wichtigsten Referenzen für dieses Buch sind folgende:

* Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani. (2021). [An Introduction to Statistical Learning: with Applications in R.](https://www.statlearning.com/) New York: Springer. 2nd Edition.
* Aurélien Géron. (2019). [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems.](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) Sebastopol: O’Reilly Media Inc. 3rd Edition.
* Christopher M. Bishop. (2006). [Pattern Recognition and Machine Learning.](https://link.springer.com/book/9780387310732) Berlin, Heidelberg: Springer.
* Kevin P. Murphy. (2012). [Machine Learning A Probabilistic Perspective.](https://mitpress.mit.edu/9780262018029/machine-learning/) The MIT Press.

Die ersten beiden Referenzen sind einführende Texte und können parallel zum vorliegenden Buch gelesen werden. Die letzten zwei Referenzen sind fortgeschrittene Texte und ich empfehle, sie erst nach dem vollständigen Verständnis des vorliegenden Buchs oder der ersten beiden Referenzen zu lesen.

## Lizenz {-}

Das vorliegende Buch ist unter Lizenz [CC BY-NC-SA 4.0 DEED](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.de) (Namensnennung, nicht-kommerziell, Weitergabe unter gleichen Bedingungen 4.0 International) lizenziert. Bitte halten Sie sich an die Lizenzbedingungen.

## Kontakt {-}

Für Fragen und Anregungen zum Buch stehe ich gerne zur Verfügung:

Martin Sterchi\
Riggenbachstrasse 16\
4600 Olten\
[martin.sterchi@fhnw.ch](mailto:martin.sterchi@fhnw.ch)





