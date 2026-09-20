# Fly-of-Wallstreet
eine fliege die ich zum traden zwinge und sie versklave damit sie mir geld macht

*(Kleingedrucktes: Die Fliege darf jederzeit „nein" sagen. NO TRADE ist ihr Ruhezustand — gezwungene Trades haben schon Apex Capital ruiniert.)*

---

## Die Idee

Eine Zucht künstlicher Fruchtfliegen, deren Gehirn nach dem Vorbild des
**Pilzkörpers** gebaut ist — des Lern- und Belohnungszentrums der Fliege, wie
es das [FlyWire-Konnektom](https://flywire.ai) (vollständige Verschaltung des
Fliegengehirns, Nature 2024, Segmentierung u. a. durch Google) zeigt.

Jede Fliege „riecht" jeden Tag den Markt (SPY), lernt durch **Dopamin**, welche
Marktlagen gut oder schlecht riechen, und entscheidet: **Long, Short oder
nichts tun**. Über viele Generationen werden die besten Fliegen gekreuzt, die
schlechten gelöscht.

```
      Markt heute                     die Fliege                          Entscheidung
 ┌──────────────────┐   ┌───────────────────────────────────────┐
 │ 15 Merkmale      │   │  30 Sinneskanäle (AN/AUS)              │
 │ Momentum, Vola,  │──▶│        │ zufällig, je 7 Eingänge        │
 │ Drawdown, VIX,   │   │        ▼                               │
 │ Zinsen …         │   │  2000 Kenyon-Zellen                    │
 └──────────────────┘   │        │ APL-Hemmung: nur 5 % feuern   │   ┌──────────┐
                        │        ▼                               │──▶│ LONG     │
                        │  GO/NOGO für Long, GO/NOGO für Short   │   │ NO TRADE │
                        │        ▲  (die EINZIGE Stelle, die    │   │ SHORT    │
                        │        │   lernt)                      │   └──────────┘
                        │     Dopamin ◀── Ergebnis nach h Tagen  │
                        └───────────────────────────────────────┘
```

### Fliege ↔ Trading

| Fliege | hier |
|---|---|
| ~50 Riechkanäle (Glomeruli) | 15 Marktmerkmale, je ein AN- und AUS-Kanal. Richtungen (Trend, Momentum) behalten ihr Vorzeichen, Niveaus (Vola, VIX) werden am letzten Jahr gemessen |
| Kenyon-Zellen mit zufälliger Verdrahtung | 2000 Zellen, jede hört auf 7 zufällige Kanäle |
| APL-Hemmneuron | nur die stärksten 5 % feuern — jede Marktlage wird ein scharfes Muster |
| Dopamin-Neuronen | Belohnung/Strafe aus dem Ergebnis nach *h* Tagen |
| Ausgangsneuronen (MBON) | GO und NOGO je Aktion; Wert = GO − NOGO |
| Lernen durch Synapsen-**Abschwächung** | Belohnung schwächt NOGO, Strafe schwächt GO, alles erholt sich langsam |
| Neuheit | Muster, zu denen sie noch kein Dopamin bekam, handelt die Fliege nie — unbekannt heißt NO TRADE |

**Warum das zum Trading passt:** Nur eine einzige Schicht lernt, der Rest ist
fest verdrahtet. Wenig Lernbares heißt wenig Raum, Rauschen auswendig zu
lernen — die Krankheit, an der fast jedes Trading-Netz stirbt. Und weil Lernen
nur abschwächt, bleiben alle Gewichte in [0, 1]; nichts kann explodieren.

## Das Belohnungssystem

`fly/dopamine.py` — hier entsteht der Charakter der Zucht:

- **Gewinn** belohnt, **Verlust** bestraft — Verlust stärker (Gen `loss_aversion`)
- Ergebnisse zählen **risikobereinigt**: 1 % im ruhigen Markt ist eine Überraschung, im Crash Rauschen
- **Kosten** (5 bp je Wechsel, 1 % p.a. Leihgebühr für Short) sind im Ergebnis schon abgezogen
- **Verpasstes tut leise weh:** Auch aus Aktionen, die sie *nicht* gemacht hat, lernt die Fliege — gedämpft durch das Gen `miss_weight`
- `tanh`-Sättigung: Ein Ausreißer-Tag kann das Gedächtnis nicht auf einen Schlag überschreiben

## Die Zucht

| | |
|---|---|
| Population | 50 Fliegen, **10 überleben** je Generation |
| Generation | ein Kalenderquartal — ~120 Generationen von 1995 bis 2024 |
| Leben | alle erleben das neue Quartal, handeln und lernen (entscheiden **vor** dem Lernen) |
| Prüfungen | dazu 2 zufällige frühere Quartale ohne Lernen |
| Fitness | `excess` (Standard): Sharpe **gegenüber Buy & Hold** über alle Prüfungstage — nur wer den Markt schlägt, überlebt. Alternativ `pooled` oder `worst` (schlechteste Prüfung zählt, v1) |
| Nachwuchs | Kreuzung zweier Überlebender + Mutation |
| Vererbung | **Gene und Gelerntes**: jedes Muster (je Kenyon-Zelle) komplett von Mutter oder Vater |
| Schädel | alle Fliegen teilen dieselbe Verdrahtung — nur so sind Erinnerungen vererbbar |
| Endprodukt | **Schwarm** der 10 Besten; gehandelt wird nur, wenn ≥ 6 zustimmen |

### Die DNA (`fly/evolution.py`)

| Gen | Bedeutung |
|---|---|
| `lr` | wie stark ein Dopamin-Stoß wirkt |
| `forget` | wie schnell Erinnerungen verblassen |
| `loss_aversion` | wie viel mehr Verlust als Gewinn zählt |
| `miss_weight` | wie laut Verpasstes zählt |
| `thr_long`, `thr_short` | wie sicher sie sein muss, bevor sie handelt — darf negativ sein, weil Verlustangst alle Werte ins Minus drückt (Short startet vorsichtiger — SPY steigt langfristig) |
| `horizon` | nach wie vielen Tagen abgerechnet wird (1–10) |
| `mut_scale` | wie stark ihre Kinder mutieren — selbst vererbt |

## Ehrlichkeit eingebaut

Evolution ist die stärkste Überanpassungsmaschine, die es gibt. Deshalb:

1. **Walk-forward:** Der Schwarm der Vorgeneration sagt jedes Quartal vorher, *bevor* es irgendwen ausliest. Nur diese Kurve zählt als Ergebnis.
2. **Kontrollversuch Zufallsauslese** (`random-selection`): Gelöscht wird zufällig statt nach Leistung. Ist das genauso gut, bringt die Evolution nichts.
3. **Kontrollversuch Zufallsdopamin** (`shuffled-dopamine`): Belohnungen stammen von zufälligen anderen Tagen. Ist das genauso gut, lernen die Fliegen nichts.
4. **Friedhof:** Ab 2025 existieren die Daten für die Evolution physisch nicht (`Market.until`). Nur der fertige Schwarm wird einmal darauf losgelassen — per `--friedhof`, bewusst sparsam.
5. **Tests** (`tests/`) sichern ab: keine Entscheidung hängt von späteren Kursen ab, Lernen geht in die richtige Richtung, Kinder erben jedes Muster ganz.

## Ergebnis v2 (2026-09-19) — die Fliegen lernen etwas Echtes, aber wenig

Walk-forward 1995–2024, Fitness `excess`, 8 Seeds (3 zum Entwickeln, 5 frische zum Prüfen):

| | p.a. | Sharpe | MaxDD | investiert |
|---|---|---|---|---|
| **echte Zucht** (frische Seeds 4–8) | +9,0 % | **0,60** | **39 %** | 83 % |
| Kontrolle Zufallsdopamin (Seeds 4–8) | +7,6 % | 0,54 | 47 % | 80 % |
| Kontrolle Zufallsauslese (Seeds 1–3) | +4,9 % | 0,41 | 45 % | 53 % |
| SPY Buy & Hold | +10,8 % | 0,63 | 55 % | 100 % |

- **Echte Belohnungen schlagen gemischte in 4 von 5 frischen Seeds** (der fünfte: Gleichstand), im Schnitt +0,06 Sharpe. Das ist der erste Beleg, dass die Fliegen etwas über den Markt lernen und nicht nur „Aktien steigen".
- **Weniger Absturz:** maximaler Einbruch 39 % statt 55 % bei Buy & Hold.
- **Noch nicht besser als Buy & Hold:** Sharpe 0,60 gegen 0,63, Rendite 9,0 % gegen 10,8 %. Der Friedhof bleibt deshalb weiter unberührt.

### Was v1 → v2 gebracht hat (und wie es gefunden wurde)

`python -m fly diagnose` misst ohne Evolution, ob eine einzelne Fliege etwas riecht
(IC = Rangkorrelation zwischen ihrem Long-Gefühl und der späteren Rendite, echt gegen gemischt):

1. **Sinne-Fehler behoben.** Alle Merkmale wurden am Jahresmittel gemessen — nach einem Jahr Aufwärtstrend hieß „knapp über dem 200er-Schnitt" plötzlich „unter normal". Richtungsmerkmale behalten jetzt ihr Vorzeichen. IC stieg, der Abstand zur Kontrolle wurde deutlich.
2. **Handelsschwellen dürfen negativ sein.** Verlustangst drückt die Werte ins Minus (Median −0,08), obwohl die Rangfolge der Tage stimmt — v1-Fliegen konnten deshalb kaum Long gehen (17 % investiert).
3. **Neuheitsdetektor**, damit negative Schwellen nicht dazu führen, dass frische Fliegen blind handeln.
4. **Langsames Vergessen** (Startbereich 5e-5–1e-3): riecht in der Diagnose klar mehr; die Evolution landet ebenfalls bei ~1e-4.
5. **Fitness gegen Buy & Hold** statt „schlechteste Prüfung": Die strenge v1-Regel belohnte Nichtstun und war die einzige Variante, bei der die Kontrolle *besser* war als die echte Zucht.

Was **nicht** hilft: die Bauweise des Schädels. 2000 vs. 5000 Kenyon-Zellen, 3–12 Eingänge, 2–10 % Aktivität — alle riechen gleich viel. Die natürliche Fliegenverdrahtung bleibt.

Was die Evolution **von selbst** gefunden hat: `miss_weight` sinkt auf 0,01–0,3 — Verpasstes soll tatsächlich nur leise wehtun.

## Ergebnis v3 (2026-09-20) — neue Sinne helfen nicht, Zeitraum schlägt Sinne

Erweiterung von `fly/senses.py`: Kreditaufschlag (HYG/LQD), Marktbreite
(RSP/SPY) und zwei Fluchtwerte, Gold und lange Anleihen (je relative Stärke
gegen SPY) — Momentum über mehrere Zeitskalen, im Stil der bestehenden
Merkmale. Schalter `--senses v2|v3` (auch `FLY_SENSES`) wählt den
Merkmalssatz; `fly/data.py` lädt die neuen ETFs nur für v3 nach, und weil
HYG erst ab 2007-04 existiert, kürzt `channels()`s `dropna()` die nutzbare
v3-Historie automatisch auf ~2009-04 (250 Tage Warmup für die längsten
Merkmale + 252 Tage rollierendes Fenster).

Weil dieser kürzere Zeitraum v3-Ergebnisse nicht mit dem v2-Ergebnis von
1995–2024 oben vergleichbar macht, misst der Vergleich beide Sinnessätze auf
demselben Fenster (`--since 2007-04-11`, nutzbar ab 2009-04 bis 2024-12).

`python -m fly diagnose` (IC = Rangkorrelation Long-Gefühl↔Rendite, Mittel
über 12 Parameterkombinationen, echt gegen gemischt):

| | IC echt | IC gemischt | Differenz |
|---|---|---|---|
| v2-Sinne | +0,031 | +0,038 | −0,008 |
| v3-Sinne | +0,038 | +0,051 | −0,013 |

`python -m fly lab --seeds 1 2 3 4 5 --controls none shuffled-dopamine`,
Walk-forward 2009-04 bis 2024-12, Fitness `excess`, Mittel über 5 Seeds:

| | p.a. | Sharpe | MaxDD | investiert |
|---|---|---|---|---|
| v2-Sinne, echte Zucht | +8,3 % | 0,60 | 34,2 % | 69,3 % |
| v2-Sinne, Kontrolle Zufallsdopamin | +11,5 % | 0,77 | 32,8 % | 88,2 % |
| v3-Sinne, echte Zucht | +7,8 % | 0,58 | 30,8 % | 72,9 % |
| v3-Sinne, Kontrolle Zufallsdopamin | +10,5 % | 0,71 | 33,8 % | 89,0 % |
| SPY Buy & Hold (gleicher Zeitraum) | +15,3 % | 0,91 | 33,7 % | 100 % |

- **Die neuen Sinne bringen nichts.** In der Diagnose liegt „echt" bei v3
  nicht klarer über „gemischt" als bei v2 — der Abstand ist sogar etwas
  negativer. In der Zucht liegen v3-Sinne bei beiden Kontrollen leicht unter
  v2-Sinnen (Sharpe −0,02 bzw. −0,06). Bei 5 Seeds ist das im Rauschen, aber
  jedenfalls kein Gewinn.
- **Überraschender Nebenbefund:** Auf 2009–2024 schlägt die Kontrolle
  Zufallsdopamin die echte Zucht deutlich — bei BEIDEN Sinnessätzen (v2 wie
  v3). Das steht im Gegensatz zum vollen 1995–2024-Ergebnis oben, wo echte
  Belohnungen die Kontrolle schlugen. Ohne den fairen Zeitraum-Vergleich hätte
  man das v2-Vollhistorie-Ergebnis fälschlich gegen ein kürzeres
  v3-Ergebnis gehalten und Unterschiede den Sinnen zugeschrieben, die in
  Wirklichkeit am Zeitraum liegen.
- Post-Finanzkrise-Bullenmarkt (2009–2024) scheint für „echtes Lernen aus
  Dopamin" schwieriger zu sein als die volle Historie seit 1995. Der Friedhof
  (ab 2025) bleibt unberührt.

## Ergebnis v1 (2026-09-19) — die Fliegen lernen noch nichts Echtes

Walk-forward 1995–2024, 50 Fliegen, 3 Seeds (`python -m fly lab`):

| | p.a. | Sharpe | MaxDD | investiert |
|---|---|---|---|---|
| **echte Zucht** (Seed 1 / 2 / 3) | +1,4 % / −0,6 % / +0,4 % | 0,23 / −0,03 / 0,09 | 47–58 % | ~17 % |
| Kontrolle Zufallsdopamin | +3,6 % / +2,0 % / +0,3 % | 0,51 / 0,33 / 0,08 | 18–27 % | 8–23 % |
| Kontrolle Zufallsauslese | ±0 % | ≈ 0 | < 9 % | < 1 % |
| SPY Buy & Hold | +10,8 % | 0,63 | 55 % | 100 % |

Was das heißt:

- **Die Evolution wirkt** — ohne Auslese bleiben die Fliegen fast immer flat.
- **Das Lernen aus echten Ergebnissen bringt aber nichts:** Mit durchgemischtem
  Dopamin sind die Fliegen gleich gut oder besser. Das Einzige, was sie
  zuverlässig lernen, ist „Aktien steigen meistens" — und dieses Signal
  überlebt auch das Mischen.
- **Short wird nie gehandelt** — der Schwarm findet nie eine 6/10-Mehrheit.
- Buy & Hold liegt meilenweit vorn. Der Friedhof bleibt deshalb unberührt:
  Es gibt noch nichts, das ihn verdient hätte.

## Loslegen

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m pytest
.venv\Scripts\python -m fly diagnose            # riecht eine einzelne Fliege überhaupt etwas?
.venv\Scripts\python -m fly evolve              # eine Zucht, Walk-forward-Ergebnis
.venv\Scripts\python -m fly lab --seeds 1 2 3   # Zucht gegen beide Kontrollversuche
.venv\Scripts\python -m fly evolve --friedhof   # + einmaliger Test auf 2025–heute
.venv\Scripts\python -m fly --senses v3 diagnose            # neue Sinne (Kredit/Breite/Flucht)
.venv\Scripts\python -m fly --senses v3 --since 2007-04-11 lab --seeds 1 2 3   # fairer Vergleich zu v2
```

Kursdaten (SPY, VIX, 10-J-Zins ab 1993) lädt der erste Lauf von Yahoo und legt
sie in `data/` ab; mit `--senses v3` zusätzlich HYG, LQD, RSP, GLD, TLT (ab
2007). `--since` schneidet von unten ab — für einen fairen Vergleich von v2-
und v3-Sinnen auf demselben Zeitraum. Jede Zucht speichert in `results/` ihren Verlauf, die
überlebenden Fliegen (`survivors.csv` mit Genen und Eltern) und den Schwarm samt
Gedächtnis (`swarm.npz`, ladbar mit `Population.load`). Eine Zucht dauert ~20 s.

## Projektaufbau

```
fly/data.py        Kursdaten, Friedhof-Schnitt
fly/senses.py      Sinneskanäle (Projektionsneuronen)
fly/skull.py       Kenyon-Zellen + APL (fest verdrahtet, geteilt)
fly/dopamine.py    Belohnungssystem
fly/population.py  lernende Ausgangsschicht, Leben, Schwarm-Abstimmung
fly/evolution.py   Gene, Kreuzung, Mutation, Auslese
fly/report.py      Kennzahlen
```
