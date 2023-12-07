# Einführung {#intro}

In diesem Kapitel geht es darum zu verstehen, was ML überhaupt ist, warum es nützlich sein kann und was typische Anwendungsfälle von ML sind. Wir werden ausserdem die verschiedenen Arten von ML kennen lernen.

## Was ist Machine Learning?

Kurze Geschichte von ML

Wie der Name sagt, geht es im ML darum, dass eine Maschine (oder präziser, ein Computer) aus einem gegebenen Datensatz automatisch lernt, ohne dass ein Mensch dem Computer (explizit) sagen muss, was er lernen soll. Der Mensch gibt jedoch dem Computer die Rahmenbedingungen für das selbständige Lernen vor. 

Bevor wir etwas konkreter anschauen, wie genau ein Computer selbständig aus Daten lernen kann, schauen wir uns die Definitionen von zwei Experten im Gebiet ML an:

::: {.rmdnote}
*"[Machine Learning is the] field of study that gives computers the ability to learn without being explicitly programmed."* Arthur Samuel, 1959

*"Machine Learning is the science (and art) of programming computers so they can learn from data."* Aurélien Géron^[Aurélien Géron. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems. Sebastopol: O’Reilly Media Inc. 3rd Edition.]
:::

Zusammenfassend lässt sich sagen, dass wir mit ML dem Computer die Möglichkeit geben, automatisch und selbständig aus Daten zu lernen. Nichtsdestotrotz braucht es Sie als ML-Expert*in, und zwar wie folgt:

1. Sie entscheiden sich für ein spezifisches ML Modell. Typischerweise kann ein ML Modell durch eine mathematische Funktion (siehe Kapitel \@ref(basics)) charakterisiert werden. ML Modelle können unterschiedlich flexibel sein und es liegt im Ermessen von Ihnen, wie flexibel das Modell sein soll. Sie müssen bei der Wahl des Modells die Komplexität des Problems berücksichtigen. Grundsätzliche gilt bei der Wahl des Modells, dass flexiblere Modelle komplexere Sachverhalte abbilden können. Ein zu flexibles Modell kann aber zu Overfitting führen, aber dazu später mehr. Dieser Schritt wird im Fachjargon typischerweise **Model Selection** (Modelauswahl) genannt.
2. Sobald Sie das Modell ausgewählt haben, übergeben Sie dem Computer (etwas vereinfacht gesagt) das Modell, einen Datensatz sowie einen Lernalgorithmus. Nun hat der Computer alle Zutaten, um automatisch zu lernen. Doch was lernt er eigentlich? Der Computer lernt die Parameter Ihres gewählten Modells, so dass das Modell sich optimal an die Daten anpasst. Dieser Schritt wird im Fachjargon **Model Training** (Trainieren des Modells) genannt.
3. Falls Sie mit dem erlernten Modell zufrieden sind, können Sie es nun entweder dazu verwenden Vorhersagen zu machen oder um Zusammenhänge in den Daten zu interpretieren und daraus wertvolle Einsichten gewinnen. Dieser Schritt wird im Fachjargon als **Model Inference** (Modellinferenz) zusammengefasst. Typischerweise sind Sie in der Realität mit dem ersten erlernten Modell allerdings noch nicht zufrieden und gehen zurück zu Schritt 1 und wählen ein anderes Modell.

Es handelt sich bei dieser Vorgehensweise um eine sehr allgemeine Beschreibung des Machine Learning Prozesses. Wie diese drei Schritte konkret funktionieren, werden Sie in den nachfolgenden Kapiteln dieses Buchs erfahren.

## Wann macht es Sinn ML einzusetzen?

Ein ML Modell zu trainieren kann viel Zeit und Geld kosten. Zum Beispiel müssen Sie unter Umständen überhaupt erst die Daten sammeln (oder von einem Datendienstleister kaufen), um ein Modell zu trainieren. Oder das Projekt ist so komplex, dass Sie als Analyst*in unzählige Stunden benötigen, um die Daten überhaupt erst in eine Form zu bringen, die es erlaubt ein Modell zu trainieren. Für neuartige DL Modelle oder Generative KI kann das Trainieren bzw. Lernen eines Modells durch den reinen Stromverbrauch bzw. die vom Cloud-Betreiber in Rechnung gestellten Kosten so hoch sein, dass sich Ihr ursprüngliches Vorhaben nicht mehr lohnt. Es ist also ungemein wichtig, dass Sie sich vor Projektbeginn gut überlegen, ob ML für Ihr vorliegendes Problem überhaupt Sinn macht und einen Mehrwert generieren kann. 

Folgende Daumenregeln^[siehe auch Seite 7 in Aurélien Géron. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems. Sebastopol: O’Reilly Media Inc. 3rd Edition.] können Ihnen dabei helfen, zu entscheiden, ob ML für Ihr Projekt Sinn macht:

* Ihr Problem entspricht einem Standard ML-Problem, das bereits mehrfach gelöst wurde und für das es sogenannte "off-the-shelf" Lösungen gibt. Beispiel: Sie wollen das Sentiment (positive vs. negative Grundhaltung) von Social Media Posts über Ihr Unternehmen automatisch klassifizieren. Dazu gibt es viele vortrainierte Modelle, die teilweise gratis verwendet werden können.
* Der manuelle Arbeitsaufwand ist sehr gross, wenn das Problem durch Menschen gelöst werden soll. Das Problem ist aber ansonsten klar strukturiert und benötigt keinen grossen kognitiven Einsatz eines Menschen. Beispiel: In den Post-Verteilzentren werden die von Hand geschriebenen PLZ problemlos mittels Computer bzw. ML Modellen "gelesen" und die Briefe und Pakete entsprechend sortiert.
* Komplexe Probleme, in denen ein Mensch keinen Überblick hat, weil so grosse und komplexe Datenmengen vorhanden sind. Wir Menschen haben grosse Mühe damit, in Rohdaten (reinen Datentabellen) irgendwelche Muster zu erkennen. In diesem Fall können wir entweder versuchen, die Daten zu visualisieren oder mithilfe von ML Zusammenhänge zu lernen, die wir sonst nicht erkennen könnten. Ein illustratives Beispiel ist das Anscombe Quartett^[https://de.wikipedia.org/wiki/Anscombe-Quartett], das vier kleine Stichproben mit jeweils elf Datenpunkten enthält. Jeder Datenpunkt wird durch eine $x$ und eine $y$ Variable beschrieben. Die vier $x$- sowie die vier $y$-Variablen haben identische Mittelwerte. Erst eine einfache Visualisierung der vier Stichproben mithilfe eines Streudiagramms zeigt die Muster sowie die Unterschiede zwischen den vier Stichproben deutlich auf. 


```
#>    x1 x2 x3 x4    y1   y2    y3    y4
#> 1  10 10 10  8  8.04 9.14  7.46  6.58
#> 2   8  8  8  8  6.95 8.14  6.77  5.76
#> 3  13 13 13  8  7.58 8.74 12.74  7.71
#> 4   9  9  9  8  8.81 8.77  7.11  8.84
#> 5  11 11 11  8  8.33 9.26  7.81  8.47
#> 6  14 14 14  8  9.96 8.10  8.84  7.04
#> 7   6  6  6  8  7.24 6.13  6.08  5.25
#> 8   4  4  4 19  4.26 3.10  5.39 12.50
#> 9  12 12 12  8 10.84 9.13  8.15  5.56
#> 10  7  7  7  8  4.82 7.26  6.42  7.91
#> 11  5  5  5  8  5.68 4.74  5.73  6.89
```

![](01-intro_files/figure-epub3/anscombe-1.png)<!-- -->

## Anwendungsfälle von ML

Stellen Sie sich vor, wir haben einen Datensatz mit 300 Spam Emails und 700 “Ham” Emails (kein Spam). Ohne Machine Learning müssten wir nun von Hand die 300 Spam Emails mit den 700 Ham Emails vergleichen und versuchen, Muster zu finden, die es uns erlauben Regeln aufzustellen, um die Spam Emails korrekt zu klassifizieren (z.B. Spam enthält tendenziell eher Geldbeträge oder Preise als Ham). Danach könnten wir die Regeln mit R implementieren. Dann stellt sich aber auch noch die Frage, wie die verschiedenen Regeln miteinander kombiniert werden, um eine Klassifikation zu machen. Dieses Vorgehen würde sehr viel zu tun geben und es würde gezwungenermassen zu willkürlichen Entscheidungen führen.

Machine Learning führt zu i) weniger Aufwand und ii) besseren Lösungen, indem wir in einem R-Skript ein Modell (z.B. logistische Regression) aufsetzen und dann dem Modell die Daten in geeigneter Form füttern. Danach lernt der Computer selbständig, wie er die Emails bestmöglich in Spam und Ham klassifiziert.

ML Beispiele
- Spam Filter
- ChatGPT
- Face Recognition in Fotos


## Supervised vs. Unsupervised Learning

Den Unterschied zwischen dem Supervised Learning und dem Unsupervised Learning können wir am besten erklären, indem wir uns mit ein paar mathematischen Grundlagen des Machine Learnings befassen. Keine Sorge, diese Grundlagen sind sehr einfach, aber versuchen Sie, diese bereits gut zu verstehen, denn wir bauen später darauf auf.

Im **Supervised Learning** haben wir einerseits sogenannte Input-Daten und andererseits einen Output, den wir vorhersagen wollen. Für die Input-Daten gibt es ganz viele verschiedene Begriffe, die synonym verwendet werden: z.B. Features, unabhängige Variablen, Attribute, Prädiktoren. Dasselbe gilt für den Output, hier gibt es folgende Synonyme: Zielvariable, abhängige Variable, Label, oder auch einfach $y$. Unsere Konvention hier ist aber folgende: es gibt Input-Daten (oder Input-Variablen) und einen Output (oder Output-Variable).

Die Input-Daten für eine Beobachtung $i$ schreiben wir mathematisch wie folgt:

$$
\mathbf{x}_i=\begin{pmatrix} x_{i1} \\ x_{i2} \\ \vdots \\ x_{ip} \end{pmatrix},
$$ 
Diese Notation bedarf ein paar Erklärungen:

* Den Index $i$ brauchen wir, um die verschiedenen Beobachtungen zu kennzeichnen. $i$ kann eine Ganzzahl zwischen $1$ und $n$ annehmen, wobei $n$ die Anzahl Beobachtungen im Datensatz bezeichnet. Wenn wir zum Beispiel etwas über die Input-Daten der dritten Beobachtung sagen wollen, dann können wir die Notation $\mathbf{x}_3$ verwenden.
* Für jede Beobachtung $i$ haben wir insgesamt $p$ Variablen, welche die verschiedenen Attribute einer Beobachtung enthalten. $x_{i1}$ bezeichnet also die erste Variable der i-ten Beobachtung, $x_{i2}$ die zweite Variable der i-ten Beobachtung und $x_{ip}$ die p-te (letzte) Variable der i-ten Beobachtung.
* Was Sie oben sehen, ist aus mathematischer Sicht ein Spaltenvektor. Im Moment reicht es, wenn Sie wissen, dass wir mit diesem Spaltenvektor die Input-Daten einer Beobachtung *kompakt* darstellen können.

Neben den Input-Daten haben wir im Supervised Learning aber wie erwähnt auch einen Output und den bezeichnen wir üblicherweise mit $y_i$. Auch hier hilft uns der Index $i$ dabei, die Beobachtungen eindeutig zu kennzeichnen. Schauen wir uns am besten kurz ein konkretes Beispiel an:

::: {.rmdtip}
**Aufgabe**

Stellen Sie sich vor, wir versuchen mithilfe eines Datensatzes von 5000 getätigten Kreditkartentransaktionen ein Modell zu trainieren, das vorhersagen kann, ob es sich bei einer gegebenen Transaktion um eine betrügerische Transaktion handelt oder nicht. Jede Transaktion in Ihrem Datensatz entspricht einer Beobachtung $i$. Der Output $y_i$ in diesem Beispiel ist eine kategorische Variable, die wir als $y_i=(\text{Betrug},\;\text{kein Betrug})$ darstellen können. Ausserdem haben Sie folgende Input-Daten:

$$
\mathbf{x}_i=\begin{pmatrix} \text{Transaktionsbetrag} \\ \text{Land des Zahlungsempfaengers} \\ \text{Zeitstempel der Transaktion} \end{pmatrix}
$$
Welche Werte nehmen in diesem Beispiel $n$ und $p$ an?
:::

**Wichtig**: Beim Supervised Learning geht es um ML Probleme, in denen sowohl Input-Daten als auch ein Output vorhanden ist. Ziel beim Supervised Learning ist es, ein Modell zu trainieren, das basierend auf den Input-Daten möglichst gute Vorhersagen für den Output macht. Es geht also hier um Vorhersageprobleme. In einem gewissen Sinn ist der Output der Überwacher (engl. Supervisor), der den Lernprozess des Modells kontrolliert.

Im Gegensatz zum Supervised Learning haben wir im **Unsupervised Learning** nur Input-Daten und *keinen Output*. Im Unsupervised Learning geht es darum, aus den Input-Daten interessante Muster zu lernen, welche für bessere unternehmerische Entscheidungen verwendet werden können. Ein einfaches Beispiel ist das Clustering von Kundinnen und Kunden eines Unternehmens in ähnliche Kundengruppen, so dass die verschiedenen Kundengruppen gezielter mit Marketingaktionen angesprochen werden können.

Neben dem Supervised und dem Unsupervised Learning gibt es noch eine dritte Kategorie von Machine Learning, nämlich das **Reinforcement Learning** (RL). Dieser Kategorie gehören Modelle an, die (virtuelle) Agenten so trainieren, dass sie langfristig möglichst optimal handeln. Das bekannteste Beispiel aus dem RL ist Googles AlphaGo Agent, welcher den menschlichen Go Weltmeister im Jahr 2017 schlug.^[https://deepmind.google/technologies/alphago/]. Reinforcement Learning ist aber auch eine wichtige Komponente in der Optimierung von grossen Sprachmodellen wie ChatGPT. In einer ersten Fassung dieses Buchs werden wir uns nicht (oder nur am Rande) mit RL befassen.

## Regression vs. Klassifikation

Im Bereich des Supervised Learnings unterscheiden wir zwischen Regressions- und Klassifikationsproblemen.

Im Regressionsproblem ist der Output bzw. die Zielvariable eine stetige Variable (Intervall- oder Verhältnisskalierung), d.h. die Variable enthält numerische Werte.

Im Klassifikationsproblem ist der Output bzw. die Zielvariable eine kategorische Variable (Nominal- oder Ordinalskalierung).

## Parametrische vs. nicht-parametrische Modelle

K-Nearest Neighbor als erstes (nicht-parametrisches) Beispiel
Hier erste kleine App

<iframe src="https://martin-sterchi.shinyapps.io/appKNN/?showcase=0" width="100%" height="600px" data-external="1"></iframe>


## Machine Learning Pipeline

Pipeline zeigen


