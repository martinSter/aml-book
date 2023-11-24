# Einführung {#intro}

In diesem Kapitel geht es darum zu verstehen, was ML überhaupt ist, warum es nützlich sein kann und was typische Anwendungsfälle von ML sind. Wir werden ausserdem die verschiedenen Arten von ML kennen lernen.

## Was ist Machine Learning?

Kurze Geschichte von ML

Wie der Name sagt, geht es im ML darum, dass eine Maschine (oder präziser, ein Computer) aus einem gegebenen Datensatz automatisch lernt, ohne dass ein Mensch dem Computer sagen muss, was er lernen soll. Der Mensch gibt jedoch dem Computer die Rahmenbedingungen für das selbständige Lernen vor. 

Bevor wir etwas konkreter anschauen, wie genau ein Computer selbständig aus Daten lernen kann, schauen wir uns die Definitionen von zwei Experten im Gebiet ML an:

*"[Machine Learning is the] field of study that gives computers the ability to learn without being explicitly programmed."* Arthur Samuel, 1959

*"Machine Learning is the science (and art) of programming computers so they can learn from data."* Aurélien Géron^[Aurélien Géron. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems. Sebastopol: O’Reilly Media Inc. 3rd Edition.]

Zusammenfassend lässt sich sagen, dass wir mit ML dem Computer die Möglichkeit geben, automatisch und selbständig aus Daten zu lernen. Nichtsdestotrotz braucht es Sie als ML-Expert*in, und zwar wie folgt:

1. Sie entscheiden sich für ein spezifisches ML Modell. Typischerweise kann ein ML Modell durch eine mathematische Funktion (Kapitel \@ref(basics)) charakterisiert werden. ML Modelle können unterschiedlich flexibel sein und es liegt im Ermessen von Ihnen, wie flexibel das Modell sein soll. Grundsätzliche gilt bei der Wahl des Modells, dass flexiblere Modelle komplexere Sachverhalte abbilden können. Ein zu flexibles Modell kann aber zu Overfitting führen, aber dazu später mehr.
2. Sobald Sie das Modell ausgewählt haben, übergeben Sie dem Computer das Modell, einen Datensatz sowie einen Lernalgorithmus. Nun hat der Computer alle Zutaten, um automatisch zu lernen, doch was lernt er eigentlich? Der Computer lernt die Parameter Ihres gewählten Modells, so dass das Modell sich optimal an die Daten anpasst.
3. Das so erlernte Modell kann nun entweder verwendet werden, um ...

## Wann macht es Sinn ML einzusetzen?

Ein ML Modell zu trainieren kann viel Zeit und Geld kosten. Zum Beispiel müssen Sie unter Umständen überhaupt erst die Daten sammeln, um ein Modell zu erlernen. Oder das Projekt ist so komplex, dass Sie als Analyst*in unzählige Stunden benötigen, um die Daten überhaupt erst in eine Form zu bringen, die es erlaubt ein Modell zu trainieren. Für neuartige DL Modelle oder Generative KI kann das Trainieren bzw. Lernen eines Modells durch den reinen Stromverbrauch bzw. die vom Cloud-Betreibeber in Rechnung gestellten Kosten so hoch sein, dass sich Ihr ursprüngliches Vorhaben nicht mehr lohnt.  Es ist also ungemein wichtig, dass Sie sich vor Projektbeginn gut überlegen, ob ML für Ihr vorliegendes Problem überhaupt Sinn macht und einen Mehrwert generieren kann. 

Folgende Daumenregeln können Ihnen dabei helfen, zu entscheiden, ob ML für Ihr Projekt Sinn macht:

* Der manuelle Arbeitsaufwand ist sehr gross, wenn das Problem durch Menschen gelöst werden soll. Das Problem ist aber ansonsten klar strukturiert und benötigt keinen grossen kognitiven Einsatz eines Menschen. Beispiel: In den Post-Verteilzentren werden die von Hand geschriebenen PLZ problemlos mittels Computer bzw. ML Modellen “gelesen” und die Briefe und Pakete entsprechend sortiert.
* Komplexe Probleme, in denen ein Mensch keinen Überblick hat, weil so viele Daten vorhanden sind (Bsp. Anscome Quartett).
* Wenn sich das Problem dynamisch verändert. ML erlaubt es uns, ein Modell effizient mit neuen Daten zu rechnen und so an das veränderte Problem anzupassen.
* ...

## Anwendungsfälle von ML

Stellen Sie sich vor, wir haben einen Datensatz mit 300 Spam Emails und 700 “Ham” Emails (kein Spam). Ohne Machine Learning müssten wir nun von Hand die 300 Spam Emails mit den 700 Ham Emails vergleichen und versuchen, Muster zu finden, die es uns erlauben Regeln aufzustellen, um die Spam Emails korrekt zu klassifizieren (z.B. Spam enthält tendenziell eher Geldbeträge oder Preise als Ham). Danach könnten wir die Regeln mit R implementieren. Dann stellt sich aber auch noch die Frage, wie die verschiedenen Regeln miteinander kombiniert werden, um eine Klassifikation zu machen. Dieses Vorgehen würde sehr viel zu tun geben und es würde gezwungenermassen zu willkürlichen Entscheidungen führen.

Machine Learning führt zu i) weniger Aufwand und ii) besseren Lösungen, indem wir in einem R-Skript ein Modell (z.B. logistische Regression) aufsetzen und dann dem Modell die Daten in geeigneter Form füttern. Danach lernt der Computer selbständig, wie er die Emails bestmöglich in Spam und Ham klassifiziert.

ML Beispiele
- Spam Filter
- ChatGPT
- Face Recognition in Fotos


## Supervised vs. Unsupervised Learning

Beim Supervised Learning geht es um ML Probleme, in denen sowohl Input-Daten als auch ein Output vorhanden ist. Für die Input-Daten gibt es ganz viele verschiedene Begriffe, die synonym verwendet werden: z.B. Features, unabhängige Variablen, Attribute, Prädiktoren. Dasselbe gilt für den Output, hier gibt es folgende Synonyme: Zielvariable, abhängige Variable, Label, oder auch einfach y.

Mathe Notation einführen.

Ziel ist es, ein Modell zu lernen, das basierend auf den Input-Daten möglichst gute Vorhersagen für den Output macht. Es geht also hier um Vorhersageprobleme. In einem gewissen Sinn ist der Output der Supervisor, der den Lernprozess des Modells überwacht.

Neben dem Supervised und dem Unsupervised Learning gibt es noch eine dritte erwähnenswerte ML Kategorie, nämlich Reinforcement Learning (RL). Dieser Kategorie gehören Modelle an, die sogenannte (virtuelle) Agenten so trainieren, dass sie langfristig möglichst optimal handeln. Das bekannteste Beispiel aus dem RL ist Googles AlphaGo Agent, welcher den menschlichen Go Weltmeister im Jahr 2017 schlug.

## Regression vs. Klassifikation

Im Bereich des Supervised Learnings unterscheiden wir zwischen Regressions- und Klassifikationsproblemen.

Im Regressionsproblem ist der Output bzw. die Zielvariable eine stetige Variable (Intervall- oder Verhältnisskalierung), d.h. die Variable enthält numerische Werte.

Im Klassifikationsproblem ist der Output bzw. die Zielvariable eine kategorische Variable (Nominal- oder Ordinalskalierung).

## Parametrische vs. nicht-parametrische Modelle

K-Nearest Neighbor als erstes (nicht-parametrisches) Beispiel
Hier erste kleine App

<iframe src="https://martin-sterchi.shinyapps.io/appKNN/?showcase=0" width="672" height="600px" data-external="1"></iframe>


## Machine Learning Pipeline

Pipeline zeigen


