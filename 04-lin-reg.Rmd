# Lineare Regression {#lin-reg}

In diesem Kapitel werden wir uns eingehend mit dem einfachsten Modell für das Regressionsproblem auseinander setzen, nämlich dem linearen Regressionsmodell. Liegt ein Regressionsproblem vor, dann macht es in der Praxis fast immer Sinn mit diesem Modell zu starten und dann die Komplexität nach Bedarf zu erhöhen.

## ML-Modelle im Allgemeinen

Wie bereits in Kapitel \@ref(intro) gesehen, geht es beim Regressionsproblem darum, eine stetige Variable $y_i \in \mathbb{R}$ möglichst optimal vorherzusagen. Dazu verwenden wir eine oder mehrere Input-Variablen, welche wir kompakt als Vektor $\mathbf{x}_i$ schreiben.

Das Problem ist nur lösbar, falls es tatsächlich einen Zusammenhang zwischen den Input-Variablen $\mathbf{x}_i$ und dem Output $y_i$ gibt. Wir nehmen ganz allgemein an, dass der Zusammenhang zwischen dem Output $y_i$ und den Input-Variablen $\mathbf{x}_i$ mathematisch wie folgt ausgedrückt werden kann:

$$
y_i = f(\mathbf{x}_i) + \epsilon
$$

* Die Funktion $f(\mathbf{x}_i)$ bezeichnet die **systematische Information**, die wir aus $\mathbf{x}_i$ im Hinblick auf $y_i$ lernen können.
* $\epsilon$ ist ein Fehlerterm, der die Differenz zwischen $y_i$ und $f(\mathbf{x}_i)$ abbildet,^[$\epsilon = y_i - f(\mathbf{x}_i)$] also den **nicht-lernbaren** (unsystematischen) **Teil**. Der Fehlerterm beinhaltet einerseits den Effekt von Variablen, die uns nicht zur Verfügung stehen, aber einen Einfluss auf den Output $y_i$ haben und andererseits nicht-messbare Variation, oft auch einfach "Noise" genannt. Grob gesagt: alles nicht-messbare.

Der Output $y_i$ ergibt sich also aus der Addition eines systematischen Teils $f(\mathbf{x}_i)$ sowie eines Fehlerterms $\epsilon$.

<div style = "background-color:#DEEBF7; padding:10px">
**Wichtig**: Ziel des Machine Learnings ist es, eine Funktion $\hat{f}(\mathbf{x}_i)$ zu trainieren (schätzen), die der wahren aber unbekannten Funktion $f(\mathbf{x}_i)$ so nahe wie möglich kommt. Im (unrealistischen) Idealfall ist unser trainiertes Modell gleich der wahren Funktion, also $\hat{f}(\mathbf{x}_i) = f(\mathbf{x}_i)$ und wir haben die systematische Information perfekt gelernt. Jedes ML-Modell, das wir uns in diesem Buch anschauen werden, kann als eine mathematische Funktion $\hat{f}(\mathbf{x}_i)$ der Input-Variablen $\mathbf{x}_i$ aufgeschrieben werden. Sobald wir $\hat{f}(\mathbf{x}_i)$ trainiert haben, können wir damit Vorhersagen machen, denn die Vorhersage für einen gegebenen Input-Vektor $\mathbf{x}_0$ ist nichts anderes als der Wert der trainierten Funktion an diesem Punkt, also $\hat{y}_0 = \hat{f}(\mathbf{x}_0)$.
</div>

## Das Modell (ausgeschrieben)

Nun wollen wir uns konkret mit dem linearen Regressionsmodell befassen. Das bedeutet nun nichts anderes, als dass wir die allgemein geschriebene Funktion $f(\mathbf{x}_i)$ durch eine konkrete mathematische Funktion ersetzen. Im Machine Learning ist das der erste wichtige Schritt, nämlich die Modellwahl (engl. *Model Selection*). Das Modell kann wie folgt geschrieben werden:

$$
f(\mathbf{x}_i) = w_0 + w_1 \cdot x_{i1} + w_2 \cdot x_{i2} + \ldots + w_p \cdot x_{ip}
$$
Wir verzichten hier bewusst darauf, den Hut für $f$ zu schreiben, da es sich lediglich um eine allgemein gültige Funktion handelt und noch nichts geschätzt bzw. trainiert wurde. Dieses Modell bzw. diese Funktion hat sogenannte **Parameter**, die es zu schätzen gilt. Hier sind dies die Parameter $w_0,\; w_1,\; \ldots,\; w_p$. Wegen der Konstante $w_0$ haben wir immer einen Parameter mehr als es Input-Variablen hat, also $p+1$ Parameter.

Diese Parameter sind die Schlüsselzutat in einem ML-Modell. Wir wollen sie optimieren, so dass die trainierte Funktion $\hat{f}(\mathbf{x}_i)$ der wahren Funktion $f(\mathbf{x}_i)$ möglichst nahe kommt.

## Das Modell (kompakt)

Sie sehen oben, dass es ziemlich umständlich sein kann, das lineare Regressionsmodell aufzuschreiben, insbesondere wenn wir viele Input-Variablen haben. Mithilfe von **Vektoren und Matrizen** können wir das Modell viel kompakter aufschreiben.

Wir haben in Kapitel \@ref(intro) bereits gesehen, dass die Input-Variablen für eine Beobachtung $i$ als Spaltenvektor geschrieben werden können. Wir modifizieren diesen Spaltenvektor in einem ersten Schritt, indem wir an erster Stelle eine 1 einfügen, also:^[So müssen wir die Konstante $w_0$ nicht separat aufschreiben.]

$$\mathbf{x}_i=\begin{pmatrix} 1\\ x_{i1} \\ x_{i2} \\ \vdots \\ x_{ip} \end{pmatrix}$$

Nun stecken wir die Parameter des Modells ebenfalls in einen Spaltenvektor:

$$\mathbf{w}=\begin{pmatrix} w_0 \\ w_1 \\ w_2 \\ \vdots \\ w_p \end{pmatrix}$$

Wir können nun das lineare Regressionsmodell (für die Beobachtung $i$) als **Skalarprodukt** dieser beiden Vektoren aufschreiben:

\begin{align}
f(\mathbf{x}_i) &= \mathbf{w}' \mathbf{x_i}\\ 
&= \begin{pmatrix} w_0 & w_1 & w_2 & \dots & w_p \end{pmatrix} \begin{pmatrix} 1\\ x_{i1} \\ x_{i2} \\ \vdots \\ x_{ip} \end{pmatrix}\\
&= w_0 \cdot 1 + w_1 \cdot x_{i1} + w_2 \cdot x_{i2} + \dots + w_p \cdot x_{ip}
\end{align}

Die Form $\mathbf{w}' \mathbf{x_i}$ ist schon ziemlich kompakt, aber es geht noch besser. Wir können nämlich das Modell gleich für alle $n$ Beobachtungen (und nicht nur für die $i$-te Beobachtung) aufschreiben. Dazu müssen wir die Input-Variablen für jede Beobachtung $i$ in einer Matrix anordnen:

$$
\mathbf{X} = \begin{pmatrix}
1 & x_{11} & x_{12} & \cdots & x_{1p}\\ 
1 & x_{21} & x_{22} & \cdots & x_{2p}\\
\vdots & \cdots & \cdots & \ddots & \vdots\\
1 & x_{n1} & x_{n2} & \cdots & x_{np}\\
\end{pmatrix}
$$

Die Matrix $\mathbf{X}$ wird typischerweise **Design Matrix** genannt. Die erste Zeile enthält die Input-Variablen für die erste Beobachtung, die zweite Zeile die Input-Variablen für die zweite Beobachtung, usw. Nun können wir das Modell mithilfe einer Multiplikation zwischen der Design Matrix $\mathbf{X}$ und dem Spaltenvektor $\mathbf{w}$ in einem Schritt für alle Beobachtungen aufschreiben:

\begin{align}
f(\mathbf{X}) &= \mathbf{X}\mathbf{w}\\
&= \begin{pmatrix}
1 & x_{11} & x_{12} & \cdots & x_{1p}\\ 
1 & x_{21} & x_{22} & \cdots & x_{2p}\\
\vdots & \cdots & \cdots & \ddots & \vdots\\
1 & x_{n1} & x_{n2} & \cdots & x_{np}\\
\end{pmatrix}\begin{pmatrix} w_0 \\ w_1 \\ w_2 \\ \dots \\ w_p \end{pmatrix}\\
&= \begin{pmatrix} 
w_0 \cdot 1 + w_1 \cdot x_{11} + w_2 \cdot x_{12} + \dots + w_p \cdot x_{1p} \\
w_0 \cdot 1 + w_1 \cdot x_{21} + w_2 \cdot x_{22} + \dots + w_p \cdot x_{2p} \\ 
\cdots \\ 
w_0 \cdot 1 + w_1 \cdot x_{n1} + w_2 \cdot x_{n2} + \dots + w_p \cdot x_{np}\end{pmatrix}
\end{align}

Überprüfen wir doch noch kurz die Dimensionen von obigem Matrix-Vektor Produkt. Die Matrix $\mathbf{X}$ hat $n$ Zeilen und $p+1$ Spalten und darum eine Dimensionalität von $n \times (p+1)$. Der Spaltenvektor $\mathbf{w}$ hat Dimensionalität $(p+1) \times 1$. Das Matrix-Vektor Produkt hat dementsprechend eine Dimensionalität von $n \times 1$, genau was wir erwarten würden, nämlich einen Vektor mit den Vorhersagen für alle $n$ Beobachtungen.

Warum wir all das tun, werden wir weiter unten sehen. Es wird unser Leben viel einfacher machen! Versuchen Sie diesen Abschnitt hier gut zu verstehen, so dass Sie sobald wie möglich mit der Matrixschreibweise von Modellen vertraut sind.

## Modelltraining

Wir werden uns hier anschauen, dass für das Training (oft auch *Fitting* genannt) des linearen Regressionsmodells zwei verschiedene Perspektiven eingenommen werden können, welche am Schluss beide zum selben Schluss kommen.

### Perspektive 1: Funktionsoptimierung

In der ersten Perspektive behandeln wir das Modelltraining als Optimierungsproblem. Wir wollen nämlich eine sogenannte **Kostenfunktion** (engl. *Loss Function*) aufstellen, die es danach zu minimieren gilt. Sie werden gleich sehen, dass die Kostenfunktion für das lineare Regressionsmodell von den Modellparameter $w_0,w_1,\dots,w_p$ abhängen wird. Das Ziel wird also sein, die optimalen Werte für die Modellparameter zu finden, so dass die Kostenfunktion so klein wie möglich ist.

Doch wie sieht denn nun diese Kostenfunktion für das lineare Regressionsmodell konkret aus? Wir werden uns hier der Einfachheit halber mal nur ein **einfaches lineares Regressionsmodell** mit nur einer Input-Variable $x_i$ anschauen. Die Kostenfunktion sieht in diesem Fall so aus:

$$
J(\hat{w}_0,\hat{w}_1) = \frac{1}{2n} \sum_{i=1}^{n} \left(y_i - \hat{f}(x_i) \right)^2
$$

Sie sehen, dass die Kostenfunktion $J(\hat{w}_0,\hat{w}_1)$ eine Funktion der beiden (trainierten) Modellparameter ist. Vielleicht wundern Sie sich nun, wie diese Kostenfunktion von den Modellparameter abhängt, da diese in obiger Formel ja gar nicht direkt ersichtlich sind. Schreiben wir die Kostenfunktion doch mal etwas um:

\begin{align}
J(\hat{w}_0, \hat{w}_1) &= \frac{1}{2n} \sum_{i=1}^{n} \left(y_i - \hat{f}(x_i) \right)^2 \\
&= \frac{1}{2n} \sum_{i=1}^{n} \left(y_i - (\hat{w}_0 + \hat{w}_1 \cdot x_i) \right)^2 \\
&= \frac{1}{2n} \sum_{i=1}^{n} \left(y_i - \hat{w}_0 - \hat{w}_1 \cdot x_i \right)^2 \\
\end{align}

Nun ist offensichtlich, wie die Kostenfunktion $J$ von den Modellparameter $\hat{w}_0$ und $\hat{w}_1$ abhängt. Im ML gibt es nun viele verschiedene Arten, wie man für die beiden Modellparameter die optimalen Werte findet. Hier ist die Lösung zum Glück einfach, denn es gibt eine sogenannte **analytische Lösung**, d.h. es ist möglich für $\hat{w}_0$ und $\hat{w}_1$ je eine Formel zu finden, die uns erlaubt die optimalen Parameterwerte auszurechnen. Die Herleitung dieser Formeln ist nicht besonders schwierig, denn wir wenden nämlich ein altbekanntes Prinzip aus der Differenzialrechnung an: Ableitung nach dem Modellparameter gleich Null setzen und nach dem Parameter auflösen.

Machen wir dies in einem ersten Schritt für $\hat{w}_0$:

\begin{align}
\frac{\partial J(\hat{w}_0, \hat{w}_1)}{\partial \hat{w}_0} &= \frac{1}{2n} \sum_{i=1}^{n} 2 \cdot \left(y_i - \hat{w}_0 - \hat{w}_1 \cdot x_i \right) \cdot (-1) \\
&= -\frac{1}{n} \sum_{i=1}^{n} \left(y_i - \hat{w}_0 - \hat{w}_1 \cdot x_i \right) \\
&= -\frac{1}{n} \sum_{i=1}^{n} y_i +  \frac{1}{n} \sum_{i=1}^{n} \hat{w}_0 + \frac{1}{n} \sum_{i=1}^{n} \hat{w}_1 \cdot x_i \\
&= -\bar{y} + \frac{1}{n} \cdot n \cdot \hat{w}_0 + \hat{w}_1 \cdot \bar{x} \\
&= -\bar{y} + \hat{w}_0 + \hat{w}_1 \cdot \bar{x}
\end{align}

Nun setzten wir die Ableitung gleich Null und lösen nach $\hat{w}_0$ auf:

\begin{align}
-\bar{y} + \hat{w}_0 + \hat{w}_1 \cdot \bar{x} &= 0 \\
\hat{w}_0 &= \bar{y} - \hat{w}_1 \cdot \bar{x}
\end{align}

Wir sehen, dass die Lösung für $\hat{w}_0$ von den beiden Mittelwerten $\bar{y}$ und $\bar{x}$ sowie von $\hat{w}_1$. Suchen wir nun also die Lösung für $\hat{w}_1$:

\begin{align}
\frac{\partial J(\hat{w}_0, \hat{w}_1)}{\partial \hat{w}_1} &= \frac{1}{2n} \sum_{i=1}^{n} 2 \cdot \left(y_i - \hat{w}_0 - \hat{w}_1 \cdot x_i \right) \cdot (-x_i) \\
&= -\frac{1}{n} \sum_{i=1}^{n} \left(y_i \cdot x_i - \hat{w}_0 \cdot x_i - \hat{w}_1 \cdot x_i^2 \right) \\
&= -\frac{1}{n} \sum_{i=1}^{n} y_i \cdot x_i + \hat{w}_0 \cdot \frac{1}{n} \sum_{i=1}^{n} x_i + \hat{w}_1 \cdot \frac{1}{n} \sum_{i=1}^{n} x_i^2 \\
&= -\frac{1}{n} \sum_{i=1}^{n} y_i \cdot x_i + \hat{w}_0 \cdot \bar{x} + \hat{w}_1 \cdot \frac{1}{n} \sum_{i=1}^{n} x_i^2 \\
\end{align}

Nun können wir wiederum die Ableitung gleich Null setzen und für $\hat{w}_0$ setzen wir unsere Lösung von oben ein. Danach lösen wir nach $\hat{w}_1$ auf:

\begin{align}
-\frac{1}{n} \sum_{i=1}^{n} y_i \cdot x_i + \hat{w}_0 \cdot \bar{x} + \hat{w}_1 \cdot \frac{1}{n} \sum_{i=1}^{n} x_i^2 &= 0 \\
(\bar{y} - \hat{w}_1 \cdot \bar{x}) \cdot \bar{x} + \hat{w}_1 \cdot \frac{1}{n} \sum_{i=1}^{n} x_i^2 &= \frac{1}{n} \sum_{i=1}^{n} y_i \cdot x_i \\
\bar{y} \cdot \bar{x} - \hat{w}_1 \cdot \bar{x}^2 + \hat{w}_1 \cdot \frac{1}{n} \sum_{i=1}^{n} x_i^2 &= \frac{1}{n} \sum_{i=1}^{n} y_i \cdot x_i \\
\hat{w}_1 \left(\frac{1}{n} \sum_{i=1}^{n} x_i^2 - \bar{x}^2 \right) &= \frac{1}{n} \sum_{i=1}^{n} y_i \cdot x_i - \bar{y} \cdot \bar{x} \\
\hat{w}_1 &= \frac{\frac{1}{n} \sum_{i=1}^{n} y_i \cdot x_i - \bar{y} \cdot \bar{x}}{\frac{1}{n} \sum_{i=1}^{n} x_i^2 - \bar{x}^2}
\end{align}

Vielleicht erkennen Sie die Ausdrücke im Zählen und Nenner der Lösung für $\hat{w}_1$: es sind dies die Kovarianz zwischen $y_i$ und $x_i$ im Zähler und die Varianz von $x_i$ im Nenner.

Yay! Nun haben wir die Formeln für die Berechnung für die optimalen Parameterwerte des einfachen linearen Regressionsmodells gefunden. Diese Methode wird **Kleinstquadratemethode** (engl. *Least squares*) genannt, weil die optimalen Parameter die Summe über die **quadrierten** Differenzen zwischen $y_i$ und den Vorhersagen $\hat{f}(x_i)$ minimiert.

<div style = "background-color:#fef9e7; padding:10px">
**Optional: Kleinstquadratemethode in Matrixform**

Die obige Herleitung funktioniert nur für das einfache lineare Regressionsmodell mit einer Input-Variable $x_i$. Wir schauen uns hier nun kurz die allgemeine Lösung in Matrixform an. Wir nehmen an, dass die Werte unseres Outputs alle in einem Spaltenvektor $\mathbf{y}$ organisiert sind und unsere Modellvorhersagen als $\mathbf{X}\mathbf{\hat{w}}$ geschrieben werden können.

Dann können wir unsere Kostenfunktion von oben wie folgt in Matrixform schreiben:

\begin{align}
J(\mathbf{\hat{w}}) &= \frac{1}{2n} (\mathbf{y} - \mathbf{X}\mathbf{\hat{w}})' (\mathbf{y} - \mathbf{X}\mathbf{\hat{w}})
\end{align}

Das sieht schlimmer aus als es ist, denn $(\mathbf{y} - \mathbf{X}\mathbf{\hat{w}})$ ist lediglich ein Spaltenvektor mit den Differenzen zwischen den wahren $y_i$ und den Vorhersagen unseres Modells. Wenn wir diesen Spaltenvektor $\mathbf{e}$ nennen, dann kann obiger Ausdruck als $\frac{1}{2n} \mathbf{e}'\mathbf{e}$ geschrieben werden, wobei $\mathbf{e}'\mathbf{e}$ ein Skalarprodukt ist und dementsprechend einen Skalar bzw. eine einzige Zahl zurück gibt. Diese Zahl multipliziert mit $\frac{1}{2n}$ ist dann nichts anderes als der Wert unserer Kostenfunktion. Sie sehen also, dass wir mit dem Skalarprodukt $\mathbf{e}'\mathbf{e}$ die Summe ersetzen können.

Nun wenden wir die bekannten Matrix-Rechenregeln an, um die Kostenfunktion umzuschreiben:

\begin{align}
J(\mathbf{\hat{w}}) &= \frac{1}{2n} (\mathbf{y} - \mathbf{X}\mathbf{\hat{w}})' (\mathbf{y} - \mathbf{X}\mathbf{\hat{w}}) \\
&= \frac{1}{2n} (\mathbf{y}' - \mathbf{\hat{w}}' \mathbf{X}') (\mathbf{y} - \mathbf{X}\mathbf{\hat{w}}) \\
&= \frac{1}{2n} (\mathbf{y}'\mathbf{y} - \mathbf{y}'\mathbf{X}\mathbf{\hat{w}} - \mathbf{\hat{w}}' \mathbf{X}'\mathbf{y} + \mathbf{\hat{w}}' \mathbf{X}'\mathbf{X}\mathbf{\hat{w}})
\end{align}

Wenn Sie sich kurz anhand der Dimensionalität der einzelnen Komponenten überlegen, was das Endprodukt des Ausdrucks $\mathbf{y}'\mathbf{X}\mathbf{\hat{w}}$ ist, dann werden Sie sehen, dass ein Skalar (Dimensionalität $1 \times 1$) resultiert. Darum muss zwingend auch die transponierte Form davon, $(\mathbf{y}'\mathbf{X}\mathbf{\hat{w}})'=\mathbf{\hat{w}}' \mathbf{X}'\mathbf{y}$ ein Skalar sein, was dazu führt, dass die beiden mittleren Terme in der letzten Zeile von obiger Kostenfunktion identisch sein müssen. Deshalb können wir die Kostenfunktion wie folgt umschreiben:

\begin{align}
J(\mathbf{\hat{w}}) &= \frac{1}{2n} (\mathbf{y}'\mathbf{y} - 2\mathbf{y}'\mathbf{X}\mathbf{\hat{w}} + \mathbf{\hat{w}}' \mathbf{X}'\mathbf{X}\mathbf{\hat{w}})
\end{align}

So, nun können wir die Kostenfunktion nach dem Spaltenvektor mit den Modellparameter $\mathbf{\hat{w}}$ ableiten. Man spricht in diesem Fall nun nicht von einer Ableitung, sondern von einem **Gradienten**. Auch die mathematische Schreibweise ist etwas anders:

\begin{align}
\nabla_{\mathbf{\hat{w}}} J(\mathbf{\hat{w}}) &= \frac{1}{2n} (- 2\mathbf{X}'\mathbf{y} + 2\mathbf{X}'\mathbf{X}\mathbf{\hat{w}}) \\
&= \frac{1}{n} (-\mathbf{X}'\mathbf{y} + \mathbf{X}'\mathbf{X}\mathbf{\hat{w}})
\end{align}

Diesen Ausdruck können wir nun wie gewohnt gleich Null setzen (wobei wir hier rechts einen Nullvektor $\mathbf{0}$ setzen) und mit den Matrix-Rechenregeln nach $\mathbf{\hat{w}}$ auflösen:

\begin{align}
\frac{1}{n} (-\mathbf{X}'\mathbf{y} + \mathbf{X}'\mathbf{X}\mathbf{\hat{w}}) &= \mathbf{0} \\
\mathbf{X}'\mathbf{X}\mathbf{\hat{w}} &= \mathbf{X}'\mathbf{y} \\
\mathbf{\hat{w}} &= (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{y}
\end{align}

**Wichtig**: Die Matrix $\mathbf{X}'\mathbf{X}$ hat eine Dimensionalität von $(p+1) \times (p+1)$, ist also quadratisch. Sie ist nur invertierbar, wenn die Design Matrix mehr Zeilen als Spalten hat, also wenn $n > (p+1)$.
</div>

### Perspektive 2: Wahrscheinlichkeitstheorie

Nun werden wir sehen, dass wir die Lösung oben (aus Perspektive 1) auch mit einer probabilistischen Sicht auf die Dinge erhalten. Dazu schreiben wir nochmals kurz den allgemein angenommenen Zusammenhang zwischen dem wahren Output $y_i$ und den Input-Variablen auf und konkretisieren ihn dann gleich für das lineare Regressionsmodell:

\begin{align}
y_i &= f(\mathbf{x}_i) + \epsilon \\
&= \mathbf{w}' \mathbf{x_i} + \epsilon \\
\end{align}

Nun nehmen wir an, dass der Fehlerterm $\epsilon$ normalverteilt ist mit Mittelwert 0 und Varianz $\sigma^2$, also $\epsilon \sim N(0,\sigma^2)$. Dies führt nun dazu, dass unser Output $y_i$ normalverteilt ist:

$$
y_i \sim N\left(\mathbf{w}' \mathbf{x_i}, \sigma^2\right)
$$

Grafisch zeigen!

Nun möchten wir wissen, was die **gemeinsame Verteilung** aller Output-Werte in unserem Datensatz ist. D.h. wie sieht die Wahrscheinlichkeit $p(y_1,y_2,\dots,y_n)$ aus? Weil wir annehmen, dass alle Beobachtungen $i$ in unserem Datensatz unabhängig sind, sieht die Antwort auf die Frage folgendermassen aus:

$$
p(y_1,y_2,\dots,y_n) = \prod_{i=1}^n N\left(\mathbf{w}' \mathbf{x_i}, \sigma^2\right)
$$

<div style = "background-color:#DEEBF7; padding:10px">
**Maximum Likelihood**

Die gemeinsame Wahrscheinlichkeit $p(y_1,y_2,\dots,y_n)$ wird in der Fachsprache **Likelihood** genannt. Die zentrale Idee hier ist, dass wir die Modellparameter $\mathbf{w}$ so wählen, dass die *Likelihood* maximal wird. Der daraus folgende Ausdruck für $\mathbf{w}$ wird **Maximum Likelihood** Schätzer genannt und oft als ML abgekürzt, was sehr verwirrlich sein kann, da wir ja auch Machine Learning so abkürzen.
</div>

Wir können nun in der Likelihood oben anstelle von $N\left(\mathbf{w}' \mathbf{x_i}, \sigma^2\right)$ jeweils die Dichtefunktion der Normalverteilung einsetzen:

\begin{align}
p(y_1,y_2,\dots,y_n) &= \prod_{i=1}^n N\left(\mathbf{w}' \mathbf{x_i}, \sigma^2\right) \\
&= \prod_{i=1}^n \frac{1}{\sigma\sqrt{2\pi}} \exp\left( -\frac{1}{2}\left(\frac{y_i - \mathbf{w}' \mathbf{x_i}}{\sigma}\right)^{\!2}\,\right)
\end{align}

Nun vollziehen wir einen kleinen mathematischen Trick, der vielfach angewendet wird: anstelle der *Likelihood* verwenden wir nun den natürlichen Logarithmus der *Likelihood*. Das ist möglich, weil sich so das Optimierungsproblem nicht verändert. Das Logarithmieren vereinfacht das Problem ungemein, denn der Logarithmus eines Produkts wird zu einer Summe der logarithmierten Elemente:

\begin{align}
\text{ln}\; p(y_1,y_2,\dots,y_n) &= \text{ln}\left(\prod_{i=1}^n \frac{1}{\sigma\sqrt{2\pi}} \exp\left( -\frac{1}{2}\left(\frac{y_i - \mathbf{w}' \mathbf{x_i}}{\sigma}\right)^{\!2}\,\right)\right) \\
&= \sum_{i=1}^n \text{ln}\left(\frac{1}{\sigma\sqrt{2\pi}} \exp\left( -\frac{1}{2}\left(\frac{y_i - \mathbf{w}' \mathbf{x_i}}{\sigma}\right)^{\!2}\,\right) \right) \\
&= \sum_{i=1}^n \text{ln}\left(1\right) - \text{ln}\left(\sigma\sqrt{2\pi}\right) - \frac{1}{2}\left(\frac{y_i - \mathbf{w}' \mathbf{x_i}}{\sigma}\right)^{\!2} \\
&= \sum_{i=1}^n \text{ln}\left(1\right) - \sum_{i=1}^n \text{ln}\left(\sigma\sqrt{2\pi}\right) - \sum_{i=1}^n \frac{1}{2}\left(\frac{y_i - \mathbf{w}' \mathbf{x_i}}{\sigma}\right)^{\!2} \\
&= n \cdot \text{ln}\left(1\right) - n \cdot \text{ln}\left(\sigma\sqrt{2\pi}\right) - \frac{1}{2\sigma^2} \sum_{i=1}^n \left(y_i - \mathbf{w}' \mathbf{x_i}\right)^{\!2}
\end{align}

Wow, nun haben wir ein tolles Resultat gefunden: je kleiner der Term $\sum_{i=1}^n \left(y_i - \mathbf{w}' \mathbf{x_i}\right)^{\!2}$ in obiger Gleichung, desto grösser ist der natürliche Logarithmus der *Likelihood*. Das heisst nichts anderes, als dass die Kleinstquadratemethode auch der *Maximum Likelihood* Schätzer ist.





## Interpretierbarkeit

Wir werden in diesem Modul sehr einfache, aber auch sehr komplexe Funktionen kennen lernen. Je komplexer die Funktionen sind, desto mehr haben wir es mit einer **Blackbox** zu tun und desto schwieriger wird es, das Modell zu interpretieren. Relativ unflexible Modelle wie z.B. die lineare Regression sind einfach interpretierbar. Die geschätzten Koeffizienten $\hat{b}_1,\; \hat{b}_2,\; \ldots$ erlauben uns direkt zu quantifizieren, was der Effekt der verschiedenen Input-Variablen ist. Im Gegensatz dazu führen komplexe Modelle (mit vielen zu optimierenden Parameter) oft zu einer sehr guten Vorhersagegüte, weil komplexe Modelle flexibel sind und darum komplexe Zusammenhänge zwischen $\mathbf{x}_i$ und $y_i$ modellieren können.

Kurz Variable Importance erwähnen

## Regularisierte Regression

Kurz auf Variable Selection (siehe ISLR) eingehen.

Hier ist nun der richtige Moment, um mal kurz auf die Theorie zum linearen Regressionsmodell **mit Regularisierung** einzugehen.

Wir haben in der Einführung gelernt, dass wir für das lineare Regressionsproblem folgende Kostenfunktion minimieren (den Mean Squared Error oder kurz MSE):

$$
J(\text{Modellparameter}) = \frac{1}{2n} \sum_{i=1}^{n} \left(y_i - \hat{f}(\mathbf{x}_i) \right)^2
$$

Normalerweise gibt es hier sogar eine analytische Lösung, d.h. es gibt Formeln wie die optimalen Parameter des Modells aus den Trainingsdaten berechnet werden können. Das ist mathematisch aber nur dann möglich, wenn die Anzahl Beobachtungen im Trainingsdatensatz grösser ist als die Anzahl Input-Variablen. Oder mathematisch ausgedrückt, nur wenn $n>p$.

Wenn $n<p$ gibt es keine analytische Lösung für das Problem. Wir haben zu wenig Datenpunkte, um die vielen Parameter des Modells schätzen zu können. Selbst wenn $n>p$, aber $p$ (Anzahl Input-Variablen) sehr gross ist, sind die Schätzungen oft nicht sehr gut, weil es dann zu Overfitting kommt. Die Lösung für dieses Problem ist **Regularisierung**.

Regularisierung bedeutet eigentlich nichts anderes, als dass wir die obige Kostenfunktion modifizieren. Dabei gibt es zwei mögliche Varianten, **Ridge** Regularisierung oder **LASSO**:

* Kostenfunktion für Ridge Regularisierung: $J(b_0,b_1,\dots,b_p) = \text{MSE} + \lambda \cdot \sum_{j=1}^p b_j^2$
* Kostenfunktion für LASSO Regularisierung: $J(b_0,b_1,\dots,b_p) = \text{MSE} + \lambda \cdot \sum_{j=1}^p |b_j|$

Diese beiden modifizierten Kostenfunktionen haben ein bisschen Erklärungsbedarf:

* In beiden Varianten wollen wir **gleichzeitig** den MSE so klein wie möglich und eine spezielle Summe über die Modellparameter (d.h. den Regularisierungsterm) so klein wie möglich machen. Das sind zwei konkurrenzierende Ziele und während des Trainings muss der beste Tradeoff gefunden werden.
* Bei Ridge ist der Regularisierungsterm eine Summe über die quadrierten Modellparameter. Das Quadrieren stellt sicher, dass sich positive und negative Parameterwerte nicht gegenseitig kompensieren.
* Bei LASSO ist der Regularisierungsterm eine Summe über die absoluten Werte der Modellparameter.
* Der Regularisierungsterm enthält die Konstante $b_0$ **nicht** (Summe startet bei $j=1$ und nicht bei $j=0$).
* Der Hyperparameter $\lambda$ legt fest, wie viel Gewicht der Regularisierungsterm bekommt. Je grösser $\lambda$, desto stärker "bestrafen" wir komplexe Modelle.
* Die Optimierung der LASSO Kostenfunktion hat einen gewichtigen Vorteil gegenüber Ridge: unwichtige Parameter werden bei LASSO automatisch auf 0 gesetzt. Das Modell nimmt also selbständig eine Selektion der wichtigen Variablen vor.

<div style = "background-color:#fef9e7; padding:10px">
**Fragen**: 

* Was passiert wenn $\lambda=0$?
* Was passiert wenn $\lambda \rightarrow \infty$?
</div><br>

Die grosse Frage ist nun, wie wir den Wert für den Hyperparameter $\lambda$ wählen. Das schauen wir uns im nächsten Abschnitt an.




## Bias-Variance Tradeoff

<div style = "background-color:#DEEBF7; padding:10px">
**Was gibt es bei der Wahl des optimalen Modells zu berücksichtigen?**

Das zentrale Thema bei der Wahl des Modells ist der **Bias-Variance Tradeoff**. Wir wollen ein Modell wählen, das weder zu viel Bias noch zu viel Varianz hat. In der Einführung zum Machine Learning haben wir bereits gelernt, dass der erwartete (quadrierte) Fehler in einen reduzierbaren und in einen nicht-reduzierbaren Fehler aufgeteilt werden kann. In der Einführung haben wir angenommen, dass $\hat{f}$ fix ist. Nun lockern wir diese Annahme und nehmen an, dass $\hat{f}$ eine Zufallsvariable ist, welche je nach Trainingsdatensatz eine unterschiedliche Form annimmt. Nach einer relativ komplizierten Herleitung kann man zeigen, dass der **erwartete quadrierte Fehler** für eine gegebene Testbeobachtung $(y_i,\mathbf{x}_i)$ wie folgt zerlegt werden kann:

$$
\text{E}\,\left[\left(y_i - \hat{f}(\mathbf{x}_i)\right)^2\right] = \text{Var}\left(\hat{f}(\mathbf{x}_i)\right) + \left[\text{Bias}\left(\hat{f}(\mathbf{x}_i)\right)\right]^2 + \text{Var}(\epsilon)
$$

Diese Formulierung beschreibt den erwarteten (quadrierten) Fehler, den wir erhalten würden, wenn wir mit einer grossen Anzahl Trainings-Datensätzen jeweils einzeln $f$ schätzen würden und dann mit Testbeobachtung $(y_i,\mathbf{x}_i)$ evaluieren würden. D.h., es ist eine ziemlich theoretische Angelegenheit, denn in der Praxis haben wir ja immer nur einen Trainingsdatensatz zur Verfügung. Aber diese Überlegungen helfen uns, das richtige Modell zu wählen.

Schauen wir uns kurz die einzelnen Komponenten auf der rechten Seite etwas genauer an:

* $\text{Var}\left(\hat{f}(\mathbf{x}_i)\right)$ misst, wie stark sich $\hat{f}$ ändert, wenn wir einen anderen Trainings-Datensatz verwenden. Ein Modell mit hoher Varianz passt sich jeweils sehr stark an die Trainingsdaten an. Je kleiner diese Varianz, desto tiefer der erwartete quadrierte Fehler.
* $\left[\text{Bias}\left(\hat{f}(\mathbf{x}_i)\right)\right]^2$ ist der quadrierte Bias und misst die systematische Abweichung vom wahren unbekannten $f$. Wir wollen natürlich auch den Bias möglichst klein halten.
* Die dritte Komponente, $\text{Var}(\epsilon)$, kennen Sie bereits. Es ist der nicht-redzierbare Fehler.

Ein Modell mit **viel Bias** führt zu einer schlechten Vorhersagequalität (auf Trainings- und Testdaten), weil das Modell zu rigide ist, um den wahren Zusammenhang zwischen der Output-Variable und den Features zu modellieren. Beispiel: wir verwenden ein einfaches lineares Regressionsmodell, um einen stark nicht-linearen Zusammenhang zwischen $x$ und $y$ zu modellieren. Im Fall von Modellen mit viel Bias spricht man auch von **Underfitting**.

Ein Modell mit **viel Varianz** führt zu einer hervorragenden Vorhersagequalität auf den Trainingsdaten, aber zu einer sehr schlechten Vorhersagequalität auf den Testdaten. Das Problem hier ist, dass das Model zu flexibel ist gemessen an der Grösse des Trainingsdatensatzes. Das Modell passt sich so zu stark an die Trainingsdaten an und modelliert auch sogenanntes **Noise** (und nicht nur das **Signal** in den Daten). Beispiel: wir modellieren ein neuronales Netzwerk, haben aber nur einen Trainingsdatensatz von einigen hundert Beobachtungen. Im Fall von Modellen mit viel Varianz spricht man auch von **Overfitting**.

Warum spricht man von einem **Tradeoff**? Flexiblere Modelle haben oft kleinen Bias, aber hohe Varianz, während unflexible Modelle oft eine kleine Varianz, aber einen hohen Bias haben. Es existiert also ein Tradeoff zwischen Bias und Varianz und wir wollen beim Modellieren und vor allem beim Hyperparameter Tuning den optimalen Tradeoff finden. (Die Intuition des Bias-Varianz Tradeoffs ist übrigens auch auf das Klassifikationsproblem übertragbar.)

**Bias-Variance Tradeoff bei Regularisierter Regression**

In unserem Beispiel wenden wir ein regularisiertes Regressionsmodell an. Hier spielt der Hyperparameter $\lambda$ (in R: `penalty`) eine zentrale Rolle für den Tradeoff zwischen Bias und Variance. Ein zu tiefer Wert für $\lambda$ kann zu einem zu flexiblen Modell mit viel Varianz führen. Ein zu hoher Wert für $\lambda$ führt zu einem zu rigiden Modell mit viel Bias.

**Wichtig:**

* Indem wir den Hyperparameter via Resampling optimieren, wählen wir automatisch ein Modell mit einem guten Tradeoff zwischen Bias und Varianz!
* Mit grossen Datensätzen ist das Problem des Overfittings weniger dramatisch. Wir haben genügend Trainingsdaten, dass selbst flexible Modelle nicht zu stark overfitten. Das ist bei unserem Beispiel der Fall (wir haben einen ziemlich grossen Trainingsdatensatz). Darum ist der optimale Hyperparameter in unserem Beispiel `penalty = 0.001`, also eher einer der kleineren Werte.

Eine letzte Überlegung bezüglich Modellselektion geht folgendermassen: wenn wir mehrere Modelle haben, die ähnlich gut performen, dann wählen wir das einfachste (kleinste) oder **am wenigsten komplexe Modell**. Man nennt dies **Occam's Razor**. William of Occam war ein Englischer Mönch und hat dieses Prinzip in einem anderen Kontext erstmals formuliert.
</div><br>


## Polynomische Regression

Wir machen hier nun einen kurzen Abstecher in die **polynomische Regression**, denn diese eignet sich sehr gut, um den Bias-Variance Tradeoff zu illustrieren.

Ein **ganz wichtiger Punkt**: das polynomische Regressionsmodell ist immer noch **linear in den Parametern**, es handelt sich also immer noch um ein lineares Modell. Sie sehen aber an obigen Modellkurven, dass dieses "lineare" Modell sehr wohl in der Lage ist, nicht-lineare Zusammenhänge zwischen $x$ und $y$ zu fitten!

## Lineare Regression in R

Base R vs. `tidymodels`


## Weiterführende Themen

Bayesianische Regression










Grob gesagt rechnen wir ein ML-Modell in zwei Schritten. In einem **ersten Schritt** entscheiden wir uns für die funktionale Form unseres Modells $\hat{f}(\mathbf{x}_i)$. Man nennt dies in der Fachsprache **Model Selection**. Wir betrachten hier nur mal den vereinfachten Fall, in dem wir nur eine $x_i$-Variable pro Beobachtung als Input haben. Folgende Funktionen bzw. Modelle sind mögliche Kandidaten:

* $f(x_i) = b_0 + b_1 \cdot x_i$ (einfache lineare Regression)
* $f(x_i) = b_0 + b_1 \cdot x_i + b_2 \cdot x_i^2$ (polynomische Regression)
* $f(x_i) = \begin{cases} \bar{y}_1, & \text{falls}\; x_i > x^*\\ \bar{y}_2, & \text{sonst} \end{cases}$

Wir werden mit unserer Wahl der Funktion nie genau die wahre aber unbekannte Funktion $f(\mathbf{x}_i)$ treffen, aber wir versuchen möglichst nahe daran zu kommen.

<div style = "background-color:#DEEBF7; padding:10px">
**"No Free Lunch" Theorem**

Das *No Free Lunch* Theorem besagt, dass es kein universal bestes Modell gibt. Das heisst, dass es je nach Problem und Datensatz andere Modelle bzw. Funktionen braucht, um gute Vorhersagen zu machen. Das ist der Hauptgrund, warum wir Ihnen möglichst viele verschiedene Tools mit auf den Weg geben wollen.
</div><br>


Im Vergleich zur Summe der quadrierten Residuen haben wir hier noch den Faktor $\frac{1}{2n}$ drin. Dieser Faktor macht daraus eine Art Mittelwert und darum wird diese Kostenfunktion typischerweise **Mean Squared Error** (MSE) genannt.









<div style = "background-color:#fef9e7; padding:10px">
**Optional: Zerlegung des Vorhersagefehlers**

Wir wollen hier kurz anschauen, wie der **Erwartungswert** des quadrierten Fehlers, $\left(y_i - \hat{f}(\mathbf{x}_i)\right)^2$, in zwei Komponenten zerlegt werden kann.

Dazu gilt folgendes:

* Von oben wissen wir, dass $y_i = f(\mathbf{x}_i) + \epsilon$ gilt.
* Wir nehmen an, dass der Erwartungswert des unsystematischen Teils $\epsilon$ Null ist, also $\text{E}(\epsilon)=0$.
* Allgemeine Regel zur Varianz einer Zufallsvariable: $\text{Var}(\epsilon) = \text{E}(\epsilon^2) - \text{E}(\epsilon)^2 = \text{E}(\epsilon^2) - 0^2 = \text{E}(\epsilon^2)$.
* $\hat{f}$ und $\mathbf{x}_i$ sind fix und gegeben (keine Zufallsvariablen) und darum gilt $\text{E}\left(\hat{f}(\mathbf{x}_i)\right)=\hat{f}(\mathbf{x}_i)$.

Nun können wir den **Erwartungswert** des quadrierten Fehlers rechnen:

\begin{align}
\text{E}\,\left[\left(y_i - \hat{f}(\mathbf{x}_i)\right)^2\right] &= \text{E}\,\left[\left(f(\mathbf{x}_i) + \epsilon - \hat{f}(\mathbf{x}_i)\right)^2\right] \\
&= \text{E}\,\left[f(\mathbf{x}_i)^2 - 2 \cdot f(\mathbf{x}_i) \cdot \hat{f}(\mathbf{x}_i) + \hat{f}(\mathbf{x}_i)^2 + 2 \cdot \epsilon \cdot f(\mathbf{x}_i) - 2 \cdot \epsilon \cdot \hat{f}(\mathbf{x}_i) + \epsilon^2 \right] \\
&= f(\mathbf{x}_i)^2 - 2 \cdot f(\mathbf{x}_i) \cdot \hat{f}(\mathbf{x}_i) + \hat{f}(\mathbf{x}_i)^2 + 2 \cdot \text{E}(\epsilon) \cdot f(\mathbf{x}_i) - 2 \cdot \text{E}(\epsilon) \cdot \hat{f}(\mathbf{x}_i) + \text{E}(\epsilon^2) \\
&= f(\mathbf{x}_i)^2 - 2 \cdot f(\mathbf{x}_i) \cdot \hat{f}(\mathbf{x}_i) + \hat{f}(\mathbf{x}_i)^2 + 2 \cdot 0 \cdot f(\mathbf{x}_i) - 2 \cdot 0 \cdot \hat{f}(\mathbf{x}_i) + \text{Var}(\epsilon) \\
&= f(\mathbf{x}_i)^2 - 2 \cdot f(\mathbf{x}_i) \cdot \hat{f}(\mathbf{x}_i) + \hat{f}(\mathbf{x}_i)^2 + \text{Var}(\epsilon) \\
&= \left(f(\mathbf{x}_i) - \hat{f}(\mathbf{x}_i)\right)^2 + \text{Var}(\epsilon)
\end{align}

Der erste Teil auf der rechten Seite der Formel beschreibt den **reduzierbaren Fehler** und der zweite Teil den **nicht-reduzierbaren Fehler**. Wir sehen also auch hier: es ist sehr wichtig, dass wir eine Funktion $\hat{f}(\mathbf{x}_i)$ schätzen, welche dem wahren funktionalen Zusammenhang $f(\mathbf{x}_i)$ möglichst nahe kommt.
</div>
