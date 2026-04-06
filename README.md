# NewMTSA

A ground-up reimplementation of **MTSA** (Modal Transition System Analyser) — a tool for the automated synthesis of controllers for reactive systems.

## About MTSA

MTSA is a research tool developed at the Universidad de Buenos Aires (UBA). It supports:

- **Controller synthesis** from environment models and goal specifications
- **Labelled Transition Systems (LTS)** modelling and analysis
- **Modal Transition Systems (MTS)** — an extension of LTS that distinguishes required from allowed behaviour
- **Fluent Linear Temporal Logic (FLTL)** for expressing safety and liveness properties
- **Discrete Event Control (DEC)** synthesis via supervisory control theory
- **Assume-Guarantee** reasoning for compositional verification

## Goals of NewMTSA

This project reimplements MTSA from scratch in clean, modern Java with the following objectives:

- Cleaner, more maintainable architecture
- Better separation of concerns between parsing, modelling, and synthesis algorithms
- Improved extensibility for new synthesis techniques
- Full test coverage

## Tech Stack

- **Language:** Java 11
- **Build tool:** Maven
- **Testing:** JUnit 5

## Project Structure

```
src/
  main/java/newmtsa/   # Source code
  test/java/newmtsa/   # Unit tests
```

## Building

```bash
mvn compile
```

## Running

```bash
mvn compile exec:java -Dexec.mainClass="newmtsa.Main"
```

Or after packaging:

```bash
mvn package
java -jar target/new-mtsa-1.0-SNAPSHOT.jar
```

## Running Tests

```bash
mvn test
```

## References

- Original MTSA tool: [MTSATool](http://mtsa.dc.uba.ar/)
- D'Ippolito, N., Fischbein, D., Chechik, M., Uchitel, S. — *MTSA: The Modal Transition System Analyser*
