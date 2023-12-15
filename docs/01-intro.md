# Einführung {#intro}

In diesem Kapitel geht es darum zu verstehen, was ML überhaupt ist, warum es nützlich sein kann und was typische Anwendungsfälle von ML sind. Wir werden ausserdem verschiedene Unterkategorien von ML kennen lernen.

## Was ist Machine Learning?

<!-- Kurze Geschichte von ML -->
<!-- Schon lange, seit 1950er / 1960er -->
<!-- Viel gemacht von Statistiker*innen -->
<!-- Computing Power -->
<!-- Algorithmische Kniffs -->
<!-- Generalization! -->

Im Prinzip geht die Geschichte des MLs weit zurück, nämlich zu den Anfängen der Statistik. Viele Modelle, die heutzutage im ML angewendet werden sind nämlich eigentlich von Statistiker\*innen erfundene Modelle. Die Geschichte des MLs und der Statistik sind darum eng verknüpft. Einen eigentlichen Startpunkt des MLs könnte man vielleicht in den 1960er Jahren ausmachen, mit den Arbeiten von Frank Rosenblatt^[https://en.wikipedia.org/wiki/Frank_Rosenblatt], welcher das sogenannte **Perceptron** und einen dazugehörigen Lernalgorithmus prägte (dazu später mehr). Danach blieb es aber rund 20 Jahre relativ ruhig bis die Forschung im Bereich Machine Learning so richtig Fahrt aufnahm. Ein grosser Schub für die Entwicklung von ML ging vom Aufkommen von extrem grossen Datenmengen (**Big Data**) und dem Internet aus. Das führte nämlich dazu, dass sich immer mehr Leute aus den Fachbereichen Informatik und Computer Science mit dem Thema ML befassten und effiziente Hard- und Software sowie algorithmische Kniffs und Tricks beisteuerten. Ausserdem ermöglichte das Internet den Zugang zu gewaltigen Datenmengen an Bildern, Videos, Klicks, etc. - denken Sie beispielsweise nur schon an die Informationen, die jede\*r von uns tagtäglich im Internet hinterlässt. Ein weiterer Schub für das Machine Learning war (und ist) zudem die immer besser werdende Rechenleistung von Computern. Diese Entwicklungen haben sich im November 2022 kulminiert in der erstmaligen breiten öffentlichen Wahrnehmung von sogenannten **Large Language Models** wie ChatGPT.

Wie der Name sagt, geht es im ML darum, dass eine Maschine (oder präziser, ein Computer) aus einem gegebenen Datensatz automatisch Muster lernt, ohne dass ein Mensch dem Computer (explizit) sagen muss, was er lernen soll. Der Mensch gibt jedoch dem Computer die Rahmenbedingungen für das selbständige Lernen vor. Die erlernten Muster sind selbstverständlich nur nützlich, wenn sie genereller Natur sind und auch für neue bzw. zukünftige Beobachtungen gelten. Beispiel: ein Spital hat während der Corona Pandemie ein Modell trainiert, um den täglichen Pflegebedarf je nach Wochentag, Saison, und weiteren Indikatoren vorherzusagen. Das Modell funktioniert nun nach der Pandemie aber nicht wunschgemäss und prognostiziert in den Tendenz einen zu hohen Pflegebedarf. Das Problem ist, dass die erlernten Muster nicht auf eine Zeit nach der Pandemie übertragbar sind. Die Trainingsdaten waren nicht repräsentativ genug.

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

Folgende Daumenregeln^[siehe auch Seiten 6 - 7 in Aurélien Géron. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems. Sebastopol: O’Reilly Media Inc. 3rd Edition.] können Ihnen dabei helfen, zu entscheiden, ob ML für Ihr Projekt Sinn macht:

* Ihr Problem entspricht einem Standard ML-Problem, das bereits mehrfach gelöst wurde und für das es sogenannte "off-the-shelf" Lösungen gibt. Beispiel: Sie wollen das Sentiment (positive vs. negative Grundhaltung) von Social Media Posts über Ihr Unternehmen automatisch klassifizieren. Dazu gibt es viele vortrainierte Modelle, die teilweise gratis verwendet werden können.
* Der manuelle Arbeitsaufwand ist sehr gross, wenn das Problem durch Menschen gelöst werden soll. Das Problem ist aber ansonsten klar strukturiert und benötigt keinen grossen kognitiven Einsatz eines Menschen. Beispiel: In den Post-Verteilzentren werden die von Hand geschriebenen PLZ problemlos mittels Computer bzw. ML Modellen "gelesen" und die Briefe und Pakete entsprechend sortiert.
* Komplexe Probleme, in denen ein Mensch keinen Überblick hat, weil so grosse und komplexe Datenmengen vorhanden sind. Wir Menschen haben grosse Mühe damit, in Rohdaten (reinen Datentabellen) irgendwelche Muster zu erkennen. In diesem Fall können wir entweder versuchen, die Daten zu visualisieren oder mithilfe von ML Zusammenhänge zu lernen, die wir sonst nicht erkennen könnten. Ein illustratives Beispiel ist das Anscombe Quartett^[https://de.wikipedia.org/wiki/Anscombe-Quartett], das vier kleine Stichproben mit jeweils elf Datenpunkten enthält. Jeder Datenpunkt wird durch eine $x$ und eine $y$ Variable beschrieben. Die vier $x$- sowie die vier $y$-Variablen haben identische Mittelwerte. Erst eine einfache Visualisierung der vier Stichproben mithilfe eines Streudiagramms zeigt die Muster sowie die Unterschiede zwischen den vier Stichproben deutlich auf. 

TODO: DL nur mit grossen Datenmengen


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
- Predictive Maintenance von Maschinen
- Fraisa Fotos
- Summarizing documents
- Customer Service Chatbots
- Credit Card Fraud Detection
- Recommender Systems
- Optimierungsprobleme (RL)
- Anomaly/Novelty Detection


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

```{=html}
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">
<html>

<head>
<style type="text/css">
body{font-family: Arial, Helvetica, Sans;}

.collapsible {
  background-color: #065535;
  color: white;
  cursor: pointer;
  padding: 18px;
  width: 100%;
  border: none;
  text-align: left;
  outline: none;
}

.active, .collapsible:hover {
  background-color: #065535;
}

.content {
  padding: 0 18px;
  display: none;
  overflow: hidden;
  background-color: #f1f1f1;
}


</style>
<meta charset="utf-8" />
<script type="text/javascript"   src="https://mathjax.rstudio.com/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"> </script>
</head>

<body>
<p>Stellen Sie sich vor, wir versuchen mithilfe eines Datensatzes von 5000 getätigten Kreditkartentransaktionen ein Modell zu trainieren, das vorhersagen kann, ob es sich bei einer gegebenen Transaktion um eine betrügerische Transaktion handelt oder nicht. Jede Transaktion in Ihrem Datensatz entspricht einer Beobachtung <span class="math inline">\(i\)</span>. Der Output <span class="math inline">\(y_i\)</span> in diesem Beispiel ist eine kategorische Variable, die wir als <span class="math inline">\(y_i=(\text{Betrug},\;\text{kein Betrug})\)</span> darstellen können. Ausserdem haben Sie folgende Input-Daten:</p>
<p><span class="math display">\[
\mathbf{x}_i=\begin{pmatrix} \text{Transaktionsbetrag} \\ \text{Land des Zahlungsempfaengers} \\ \text{Zeitstempel der Transaktion} \end{pmatrix}
\]</span></p>
<p>Welche Werte nehmen in diesem Beispiel <span class="math inline">\(n\)</span> und <span class="math inline">\(p\)</span> an?</p>
<ol type="a">
<li>
<span class="math inline">\(n=100\)</span> und <span class="math inline">\(p=3\)</span>
</li>
<li>
<span class="math inline">\(n=5000\)</span> und <span class="math inline">\(p=3\)</span>
</li>
<li>
<span class="math inline">\(n=3\)</span> und <span class="math inline">\(p=5000\)</span>
</li>
<li>
<span class="math inline">\(n=100\)</span> und <span class="math inline">\(p=4\)</span>
</li>
</ol>
<br/>

<button type="button" class="collapsible">Lösung</button>
<div class="content">
<br/>
  <p>Wir haben 5000 Beobachungen, darum gilt <span class="math inline">\(n=5000\)</span>. Ausserdem haben wir 3 Variablen (Attribute), darum gilt <span class="math inline">\(p=3\)</span>.</p>
<ol type="a">
<li>
Falsch
</li>
<li>
Richtig
</li>
<li>
Falsch
</li>
<li>
Falsch
</li>
</ol>
</div>

</body>
</html>
```

:::

**Wichtig**: Beim Supervised Learning geht es um ML Probleme, in denen sowohl Input-Daten als auch ein Output vorhanden ist. Ziel beim Supervised Learning ist es, ein Modell zu trainieren, das basierend auf den Input-Daten möglichst gute Vorhersagen für den Output macht. Es geht also hier um Vorhersageprobleme. In einem gewissen Sinn ist der Output die überwachend Instanz (engl. Supervisor), welche den Lernprozess des Modells kontrolliert.

Im Gegensatz zum Supervised Learning haben wir im **Unsupervised Learning** nur Input-Daten und *keinen Output*. Im Unsupervised Learning geht es darum, aus den Input-Daten interessante Muster zu lernen, welche für bessere unternehmerische Entscheidungen verwendet werden können. Ein einfaches Beispiel ist das Clustering von Kundinnen und Kunden eines Unternehmens in ähnliche Kundengruppen, so dass die verschiedenen Kundengruppen gezielter mit Marketingaktionen angesprochen werden können. Techniken, um komplexe Datensätze zu visualisieren, werden typischerweise auch zum Unsupervised Learning gezählt.

Neben dem Supervised und dem Unsupervised Learning gibt es noch eine dritte Kategorie von Machine Learning, nämlich das **Reinforcement Learning** (RL). Dieser Kategorie gehören Modelle an, die (virtuelle) Agenten so trainieren, dass sie langfristig möglichst optimal handeln. Das bekannteste Beispiel aus dem RL ist Googles AlphaGo Agent, welcher den menschlichen Go Weltmeister im Jahr 2017 schlug.^[https://deepmind.google/technologies/alphago/]. Reinforcement Learning ist aber auch eine wichtige Komponente in der Optimierung von grossen Sprachmodellen wie ChatGPT. In einer ersten Fassung dieses Buchs werden wir uns nicht (oder nur am Rande) mit RL befassen.

Die Unterscheidung zwischen den drei Arten von Machine Learning ist im oberen Teil der Abbildung \@ref(fig:slulrl) visualisiert:

<div class="figure" style="text-align: center">
<img src="images/SL_UL_RL.png" alt="Die verschiedenen Kategorien des Machine Learnings und deren Hierarchie. " width="90%" />
<p class="caption">(\#fig:slulrl)Die verschiedenen Kategorien des Machine Learnings und deren Hierarchie. </p>
</div>

## Regression vs. Klassifikation

In der Kategorie des Supervised Learnings unterscheiden wir weiter zwischen Regressions- und Klassifikationsproblemen (siehe auch Abbildung \@ref(fig:slulrl)).

Im Regressionsproblem ist der Output eine **stetige** Variable (Intervall- oder Verhältnisskalierung), d.h. die Variable enthält numerische Werte.

Im Klassifikationsproblem ist der Output bzw. die Zielvariable eine **kategorische** Variable (Nominal- oder Ordinalskalierung).

::: {.rmdtip}
**Aufgabe**

```{=html}
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">
<html>

<head>
<style type="text/css">
body{font-family: Arial, Helvetica, Sans;}

.collapsible {
  background-color: #065535;
  color: white;
  cursor: pointer;
  padding: 18px;
  width: 100%;
  border: none;
  text-align: left;
  outline: none;
}

.active, .collapsible:hover {
  background-color: #065535;
}

.content {
  padding: 0 18px;
  display: none;
  overflow: hidden;
  background-color: #f1f1f1;
}

</style>
<meta charset="utf-8" />
<script type="text/javascript"   src="https://mathjax.rstudio.com/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"> </script>
</head>

<body>
<p>Welche der folgenden Probleme sind Regressionsprobleme?</p>
<ol type="a">
<li>
Vorhersage des Lohns der Leiter*in eines Unternehmens basierend auf Profit, Marktkapitalisation, Anzahl Mitarbeitender, sowie Sektor, in dem das Unternehmen tätig ist.
</li>
<li>
Basierend auf der aktuellen Marktlage und weiteren wirtschaftlichen Aspekten wollen Sie den morgigen Preis einer bestimmten Aktie vorhersagen
</li>
<li>
Vorhersage des Alters einer Person, welche ein bestimmtes Youtube Video schauen will.
</li>
<li>
Eine Bank möchte mithilfe von historischen Daten vorhersagen, ob ein bestimmter Kunde zahlungsunfähig wird oder nicht.
</li>
<li>
Ein Detailhandelsunternehmen möchte vorhersagen, ob eine Kundin ein Produkt aus der Kategorie A, B, C, oder kein Produkt kauft.
</li>
<li>
Vorhersage von Hauspreisen basierend auf Attributen wie der Grösse, Anzahl Zimmer, Seeblick (ja/nein), Steuerlast, etc.
</li>
<li>
Ein Unternehmen lanciert ein neues Produkt und schätzt anhand von Konkurrenzprodukten, ob das eigene Produkt ein Erfolg wird oder nicht.
</li>
</ol>
<br/>

<button type="button" class="collapsible">Lösung</button>
<div class="content">
<br/>
<ol type="a">
<li>
Richtig
</li>
<li>
Richtig
</li>
<li>
Richtig
</li>
<li>
Falsch
</li>
<li>
Falsch
</li>
<li>
Richtig
</li>
<li>
Falsch
</li>
</ol>
</div>

<script>
var coll = document.getElementsByClassName("collapsible");
var i;

for (i = 0; i < coll.length; i++) {
  coll[i].addEventListener("click", function() {
    this.classList.toggle("active");
    var content = this.nextElementSibling;
    if (content.style.display === "block") {
      content.style.display = "none";
    } else {
      content.style.display = "block";
    }
  });
}
</script>

</body>
</html>
```

:::

## Parametrische vs. nicht-parametrische Modelle

Ein ML Modell gehört entweder der Familie **parametrischer** Modelle oder der Familie **nicht-parametrischer** Modelle an. Dabei spielt es keine Rolle, ob wir mit dem Modell ein Regressions- oder ein Klassifikationsproblem lösen wollen.

Womöglich sind Sie in Ihrer Ausbildung bereits **parametrischen Modellen** begegnet, denn das einfache lineare Regressionsmodell ist ein typisches Beispiel für ein parametrisches ML Modell. Das Modell ist vollkommen charakterisiert durch die beiden lernbaren (optimierbaren) Parameter $w_0$ und $w_1$^[In Statistikvorlesungen werden die beiden Parameter oft eher mit $b_0$ und $b_1$ oder mit $\beta_0$ und $\beta_1$ bezeichnet. Im Machine Learning nennt man Parameter oft Gewichte (engl. Weights), weshalb die Parameter typischerweise mit $w$ bezeichnet werden.] und kann wie folgt (mathematisch) aufgeschrieben werden:

$$
\hat{y_i} = f(x_i)=w_0 + w_1 \cdot x_i
$$
Wenn Ihnen der obige Ausdruck noch fremd vorkommt, dann ist das nicht schlimm. Wir werden im Kapitel \@ref(lin-reg) ausführlich auf lineare Regressionsmodelle eingehen. Im Moment müssen Sie nur wissen, dass ein parametrisches Modell wie oben mit einer mathematischen Funktion beschrieben werden kann und dass diese Funktion durch lernbare **Parameter** (hier $w_0$ und $w_1$) charakterisiert wird.

**Nicht-parametrische Modelle** wiederum sind Modelle, welche nicht (oder zumindest nicht explizit) durch Parameter charakterisiert sind. Am besten schauen wir uns gleich ein einfaches nicht-parametrisches Modell an, nämlich das **K-Nearest-Neighbors** (KNN) Modell. Stellen Sie sich vor, Sie haben einen Datensatz mit 55 Produkten aus Ihrem Sortiment. Sie haben jedes dieser 55 Produkte auf Instagram und auf Tiktok durch Influencer\*innen bewerben lassen. Für jedes der 55 Produkte hatten Sie ein Werbebudget für Instagram ($x_{i1}$) und ein Werbebudget für Tiktok ($x_{i2}$). Am Ende des Geschäftsjahrs haben Sie für jedes der 55 Produkte bestimmt, ob die Absatzziele erreicht wurden oder nicht (Output $y_i$). Die erfolgreichen Produkte (= Absatzziel erreicht) sind in untenstehender App als blaue Punkte eingezeichnet. Die roten Dreiecke repräsentieren die nicht-erfolgreichen Produkte. Sie sehen, dass erfolgreiche Produkte tendenziell höhere Instagram und Tiktok Werbebudgets aufwiesen als nicht-erfolgreiche Produkte. Sie möchten nun ein Modell schätzen, dass die Produkte automatisch klassifizieren kann. Dazu verwenden Sie das KNN Modell, das die $K$ nächsten Nachbarn unter den 55 gegebenen Produkten sucht und dann die häufigste Beobachtung unter den $K$ nächsten Nachbarn vorhersagt. In anderen Worten: wir suchen die $K$ **ähnlichsten** Beobachtungen und nutzen diese, um eine Vorhersage zu machen.

Selbstverständlich spielt der konkrete Wert von $K$ hier eine grosse Rolle - sollen wir nur $K=1$ Nachbarn berücksichtigen? Oder $K=10$ Nachbarn? Die erste Abbildung in der App zeigt nicht nur die 55 Datenpunkte, sondern auch die **Entscheidungsgrenze** (in schwarz). Untersuchen Sie kurz, wie sich diese Entscheidungsgrenze verändert, wenn Sie $K$ erhöhen oder reduzieren.

Ausserdem können Sie in der ersten Abbildung auch den schwarzen Punkt mit der Maus setzen, wodurch Ihnen die $K$ nächsten Punkte des schwarzen Punkts angezeigt werden.

Die zweite Abbildung zeigt die Entscheidungsregionen mit unterschiedlicher Intensität je nachdem wie sicher sich das Modell ist. In einer Region, in der alle $K$ Nachbarn nicht-erfolgreiche Produkte sind, sind wir uns eher sicher bezüglich der Vorhersage als in einer Region, in der die Anteile zwischen erfolgreichen und nicht-erfolgreichen Produkten ausgeglichen sind.

<iframe src="https://martin-sterchi.shinyapps.io/appKNN/?showcase=0" width="100%" height="600px" data-external="1"></iframe>

Um die $K$ nächsten Nachbarn zu finden, müssen wir die Distanzen zwischen Punkten rechnen können. Dazu verwenden wir die Euklidische Distanz, welche wir in Kapitel \@ref(basics) kennen lernen werden.

Das KNN Modell ist ein sehr einfaches ML Modell, welches in der Praxis allerdings nicht allzu häufig angewendet wird. Warum nicht? Weil es am sogenannten **Fluch der Dimensionalität** (engl. Curse of Dimensionality) leidet. Doch was bedeutet das? Je mehr Input-Variablen wir haben, desto weiter entfernt sind Datenpunkte voneinander (das ist etwas, das man sich nur schwer vorstellen kann, aber Sie können es mir für den Moment einfach mal glauben). Das KNN beruht auf der Grundidee, dass wir $K$ nahe, ähnliche Beobachtungen für die Vorhersage verwenden. Wenn diese $K$ nahen Beobachtungen im hochdimensionalen Raum (= viele Input-Variablen) nicht mehr nahe sind, dann funktioniert auch das Modell nicht mehr gut.

Ev. Market-Basket Analyse

Zwei Fragen mit Bildern (siehe Rapp)


## Machine Learning Pipeline

Pipeline zeigen


