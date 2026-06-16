import type {ReactNode} from 'react';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import BrowserOnly from '@docusaurus/BrowserOnly';
import benchmarks from '@site/src/data/benchmarks.json';

// Recharts has client-only code paths; wrap in BrowserOnly to avoid SSR errors.
function BenchmarkChart() {
  return (
    <BrowserOnly fallback={<div style={{height: 320}} />}>
      {() => {
        const {
          BarChart,
          Bar,
          XAxis,
          YAxis,
          Tooltip,
          ResponsiveContainer,
          CartesianGrid,
          Cell,
        } = require('recharts');

        const data = benchmarks.qeComparison.map((row) => ({
          name: row.method.replace(/, ecut=.*$/, ''),
          delta: Math.abs(row.deltaMevPerAtom),
        }));

        return (
          <div style={{width: '100%', height: 320}}>
            <ResponsiveContainer>
              <BarChart
                data={data}
                margin={{top: 16, right: 24, left: 8, bottom: 32}}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="var(--ifm-color-emphasis-300)" />
                <XAxis dataKey="name" tick={{fontSize: 12}} interval={0} />
                <YAxis
                  scale="log"
                  domain={[0.01, 100]}
                  tick={{fontSize: 12}}
                  label={{
                    value: '|ΔE| (meV/atom, log)',
                    angle: -90,
                    position: 'insideLeft',
                    style: {textAnchor: 'middle', fontSize: 12},
                  }}
                />
                <Tooltip
                  formatter={(v: number) => [`${v.toFixed(3)} meV/atom`, '|KRONOS − QE|']}
                />
                <Bar dataKey="delta" fill="var(--ifm-color-primary)">
                  {data.map((_, i) => (
                    <Cell key={i} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        );
      }}
    </BrowserOnly>
  );
}

function HeadlineHero() {
  return (
    <section
      style={{
        background: 'linear-gradient(135deg, var(--ifm-color-primary-darker), var(--ifm-color-primary-darkest))',
        color: 'white',
        padding: '4rem 1rem',
        textAlign: 'center',
      }}>
      <div className="container">
        <Heading as="h1" style={{color: 'white', marginBottom: '0.5rem'}}>
          Validated against Quantum ESPRESSO
        </Heading>
        <p style={{fontSize: '1.1rem', opacity: 0.9, marginBottom: '3rem'}}>
          Real numbers from the regression suite. Reproducible.
        </p>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
            gap: '2rem',
            maxWidth: 1000,
            margin: '0 auto',
          }}>
          {benchmarks.headline.map((h, i) => (
            <div key={i}>
              <div style={{fontSize: '2.5rem', fontWeight: 800, lineHeight: 1.1}}>
                {h.value}
                <span style={{fontSize: '1rem', fontWeight: 400, marginLeft: '0.4rem', opacity: 0.85}}>
                  {h.unit}
                </span>
              </div>
              <div style={{fontSize: '1.05rem', fontWeight: 600, marginTop: '0.4rem'}}>
                {h.label}
              </div>
              <div style={{fontSize: '0.85rem', opacity: 0.8, marginTop: '0.3rem'}}>
                {h.context}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Methodology() {
  return (
    <section style={{padding: '2.5rem 1rem'}}>
      <div className="container" style={{maxWidth: 900}}>
        <Heading as="h2">Methodology</Heading>
        <p>
          Each comparison uses the <strong>same pseudopotential file</strong>, the{' '}
          <strong>same cell parameters</strong>, and the <strong>same plane-wave cutoff</strong>{' '}
          on both KRONOS and Quantum ESPRESSO (QE) v7.x. Total energies are compared in
          Rydberg, with the per-atom delta reported in meV. The Δ-test target for production
          GPU/HPC DFT codes is &lt; 2 meV/atom; KRONOS is comfortably inside that on Si LDA
          across multiple k-grids.
        </p>
        <p>
          The Si Γ-only and 4×4×4 numbers below come from the project regression suite
          (<code>test_validation.cpp::QEValidation.*</code>). Forces are validated against
          finite differences of the total energy on the same configurations.
        </p>
        <p style={{
          borderLeft: '4px solid var(--ifm-color-primary)',
          background: 'var(--ifm-color-emphasis-100)',
          padding: '0.75rem 1rem',
          marginTop: '1.5rem',
          fontSize: '0.95rem',
        }}>
          <strong>Quantum ESPRESSO reference values</strong> on this page were obtained by
          running QE v7.x (<code>pw.x</code>) on the matching input decks. The QE input
          files are derived from QE's distributed <code>PW/examples/example01</code>{' '}
          (<em>scf-gamma</em> and <em>scf-kauto</em>) using the same{' '}
          <code>Si.pz-vbc.UPF</code> pseudopotential. Inputs are reproduced under{' '}
          <code>benchmarks/</code> (work in progress); pull requests adding more
          systems are welcome.
        </p>
      </div>
    </section>
  );
}

function QEComparisonTable() {
  return (
    <section style={{padding: '0 1rem 2.5rem'}}>
      <div className="container" style={{maxWidth: 1100}}>
        <Heading as="h2">Total energy vs QE</Heading>
        <div style={{overflowX: 'auto'}}>
          <table style={{width: '100%', borderCollapse: 'collapse'}}>
            <thead>
              <tr>
                <th style={{textAlign: 'left'}}>System</th>
                <th style={{textAlign: 'left'}}>Method</th>
                <th style={{textAlign: 'right'}}>KRONOS (Ry)</th>
                <th style={{textAlign: 'right'}}>QE (Ry)</th>
                <th style={{textAlign: 'right'}}>|Δ| (meV/atom)</th>
              </tr>
            </thead>
            <tbody>
              {benchmarks.qeComparison.map((row, i) => (
                <tr key={i}>
                  <td>{row.system}</td>
                  <td>
                    {row.method}
                    {row.note && (
                      <div style={{fontSize: '0.85rem', opacity: 0.7}}>{row.note}</div>
                    )}
                  </td>
                  <td style={{textAlign: 'right', fontFamily: 'var(--ifm-font-family-monospace)'}}>
                    {row.kronos.toFixed(6)}
                  </td>
                  <td style={{textAlign: 'right', fontFamily: 'var(--ifm-font-family-monospace)'}}>
                    {row.qe.toFixed(6)}
                  </td>
                  <td style={{textAlign: 'right', fontFamily: 'var(--ifm-font-family-monospace)', fontWeight: 600}}>
                    {row.deltaMevPerAtom.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

function ChartSection() {
  return (
    <section style={{padding: '0 1rem 2.5rem'}}>
      <div className="container" style={{maxWidth: 1100}}>
        <Heading as="h2">Per-configuration error</Heading>
        <p style={{opacity: 0.85, marginBottom: '1rem'}}>
          Logarithmic scale. The dashed line at 2 meV/atom marks the Δ-test target for
          production DFT codes.
        </p>
        <BenchmarkChart />
      </div>
    </section>
  );
}

function ComponentBreakdown() {
  const cb = benchmarks.componentBreakdown;
  return (
    <section style={{padding: '0 1rem 2.5rem'}}>
      <div className="container" style={{maxWidth: 900}}>
        <Heading as="h2">Energy components: {cb.system}</Heading>
        <p style={{opacity: 0.85}}>
          Breaking down the total energy into Hartree, XC, kinetic-plus-local, and Ewald
          contributions exposes any single component that diverges — useful for ruling out
          a "fortunate cancellation."
        </p>
        <table style={{width: '100%', borderCollapse: 'collapse'}}>
          <thead>
            <tr>
              <th style={{textAlign: 'left'}}>Component</th>
              <th style={{textAlign: 'right'}}>KRONOS (Ry)</th>
              <th style={{textAlign: 'right'}}>QE (Ry)</th>
              <th style={{textAlign: 'right'}}>Δ (Ry)</th>
            </tr>
          </thead>
          <tbody>
            {cb.rows.map((row, i) => (
              <tr key={i} style={row.component === 'TOTAL' ? {fontWeight: 700, borderTop: '2px solid var(--ifm-color-emphasis-400)'} : undefined}>
                <td>{row.component}</td>
                <td style={{textAlign: 'right', fontFamily: 'var(--ifm-font-family-monospace)'}}>{row.kronos.toFixed(5)}</td>
                <td style={{textAlign: 'right', fontFamily: 'var(--ifm-font-family-monospace)'}}>{row.qe.toFixed(5)}</td>
                <td style={{textAlign: 'right', fontFamily: 'var(--ifm-font-family-monospace)'}}>{row.delta}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ForceValidation() {
  return (
    <section style={{padding: '0 1rem 2.5rem'}}>
      <div className="container" style={{maxWidth: 900}}>
        <Heading as="h2">Force validation (Hellmann-Feynman vs finite difference)</Heading>
        <p style={{opacity: 0.85}}>
          Si diamond, atom displaced 0.01 fractional units, ecut=12 Ry. Analytic
          forces match the finite-difference gradient of the total energy to 5
          significant figures.
        </p>
        <table style={{width: '100%', borderCollapse: 'collapse'}}>
          <thead>
            <tr>
              <th style={{textAlign: 'left'}}>Component</th>
              <th style={{textAlign: 'right'}}>Analytic (Ry/bohr)</th>
              <th style={{textAlign: 'right'}}>FD (Ry/bohr)</th>
              <th style={{textAlign: 'right'}}>|Δ|</th>
            </tr>
          </thead>
          <tbody>
            {benchmarks.forceValidation.map((row, i) => (
              <tr key={i} style={row.component === 'Total' ? {fontWeight: 700, borderTop: '2px solid var(--ifm-color-emphasis-400)'} : undefined}>
                <td>{row.component}</td>
                <td style={{textAlign: 'right', fontFamily: 'var(--ifm-font-family-monospace)'}}>{row.analytic.toFixed(6)}</td>
                <td style={{textAlign: 'right', fontFamily: 'var(--ifm-font-family-monospace)'}}>{row.fd.toFixed(6)}</td>
                <td style={{textAlign: 'right', fontFamily: 'var(--ifm-font-family-monospace)'}}>{row.diff}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function MultiSystemTable() {
  return (
    <section style={{padding: '0 1rem 2.5rem'}}>
      <div className="container" style={{maxWidth: 1100}}>
        <Heading as="h2">Multi-system regression suite</Heading>
        <p style={{opacity: 0.85}}>
          Every commit runs the full regression suite across these systems. The values
          below are the converged total energies — the same numbers the test asserts on
          every CI build.
        </p>
        <div style={{overflowX: 'auto'}}>
          <table style={{width: '100%', borderCollapse: 'collapse'}}>
            <thead>
              <tr>
                <th style={{textAlign: 'left'}}>System</th>
                <th style={{textAlign: 'left'}}>Pseudopotential</th>
                <th style={{textAlign: 'left'}}>Method</th>
                <th style={{textAlign: 'right'}}>E (Ry)</th>
                <th style={{textAlign: 'left'}}>Notes</th>
              </tr>
            </thead>
            <tbody>
              {benchmarks.multiSystem.map((row, i) => (
                <tr key={i}>
                  <td>{row.system}</td>
                  <td style={{fontFamily: 'var(--ifm-font-family-monospace)', fontSize: '0.9rem'}}>{row.pp}</td>
                  <td>{row.method}</td>
                  <td style={{textAlign: 'right', fontFamily: 'var(--ifm-font-family-monospace)'}}>
                    {row.energy === null ? '—' : row.energy.toFixed(3)}
                  </td>
                  <td style={{fontSize: '0.9rem', opacity: 0.85}}>{row.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

function Citations() {
  const qeBibtex = `@article{Giannozzi2009,
  title = {QUANTUM ESPRESSO: a modular and open-source software project for
           quantum simulations of materials},
  author = {Giannozzi, P. and Baroni, S. and Bonini, N. and Calandra, M. and
            Car, R. and Cavazzoni, C. and Ceresoli, D. and Chiarotti, G. L. and
            Cococcioni, M. and Dabo, I. and others},
  journal = {Journal of Physics: Condensed Matter},
  volume = {21},
  number = {39},
  pages = {395502},
  year = {2009},
  publisher = {IOP Publishing},
  doi = {10.1088/0953-8984/21/39/395502}
}

@article{Giannozzi2017,
  title = {Advanced capabilities for materials modelling with Quantum ESPRESSO},
  author = {Giannozzi, P. and Andreussi, O. and Brumme, T. and others},
  journal = {Journal of Physics: Condensed Matter},
  volume = {29},
  number = {46},
  pages = {465901},
  year = {2017},
  publisher = {IOP Publishing},
  doi = {10.1088/1361-648X/aa8f79}
}`;

  return (
    <section style={{padding: '0 1rem 2.5rem', borderTop: '1px solid var(--ifm-color-emphasis-300)', marginTop: '2rem'}}>
      <div className="container" style={{maxWidth: 900}}>
        <Heading as="h2">Citations and references</Heading>
        <p>
          KRONOS's reference comparison code is{' '}
          <a href="https://www.quantum-espresso.org/" target="_blank" rel="noreferrer">
            Quantum ESPRESSO
          </a>{' '}
          (QE), a widely used open-source plane-wave DFT engine. If you use the numbers on
          this page in a publication, please cite both KRONOS (forthcoming, see{' '}
          <a href="/docs/user_guide">/docs</a>) and the QE primary references:
        </p>
        <ul>
          <li>
            Giannozzi, P. <em>et al.</em>{' '}
            "<a href="https://doi.org/10.1088/0953-8984/21/39/395502" target="_blank" rel="noreferrer">
              QUANTUM ESPRESSO: a modular and open-source software project for quantum
              simulations of materials
            </a>".{' '}
            <em>J. Phys.: Condens. Matter</em> <strong>21</strong>, 395502 (2009).
          </li>
          <li>
            Giannozzi, P. <em>et al.</em>{' '}
            "<a href="https://doi.org/10.1088/1361-648X/aa8f79" target="_blank" rel="noreferrer">
              Advanced capabilities for materials modelling with Quantum ESPRESSO
            </a>".{' '}
            <em>J. Phys.: Condens. Matter</em> <strong>29</strong>, 465901 (2017).
          </li>
          <li>
            Quantum ESPRESSO website:{' '}
            <a href="https://www.quantum-espresso.org/" target="_blank" rel="noreferrer">
              quantum-espresso.org
            </a>{' '}
            (releases &amp; download)
          </li>
          <li>
            QE source repository:{' '}
            <a href="https://gitlab.com/QEF/q-e" target="_blank" rel="noreferrer">
              gitlab.com/QEF/q-e
            </a>
          </li>
        </ul>
        <p>BibTeX for QE:</p>
        <pre style={{fontSize: '0.85rem'}}>
          <code>{qeBibtex}</code>
        </pre>
        <p style={{fontSize: '0.9rem', opacity: 0.8}}>
          The pseudopotential <code>Si.pz-vbc.UPF</code> used throughout the Si comparisons
          ships with the QE distribution (PZ LDA, norm-conserving, Z<sub>val</sub>=4).
        </p>
      </div>
    </section>
  );
}

function Reproducibility() {
  return (
    <section style={{padding: '0 1rem 4rem'}}>
      <div className="container" style={{maxWidth: 900}}>
        <Heading as="h2">Reproduce these numbers</Heading>
        <p style={{opacity: 0.85}}>
          The regression suite that produces every number on this page lives in the
          repo: <code>test/test_validation.cpp</code>. Build KRONOS, run{' '}
          <code>ctest -R QEValidation</code>, and the matching QE input decks for
          side-by-side runs are at <code>benchmarks/</code> (work in progress).
        </p>
        <pre>
          <code>{`# Build
cmake -B build -S . && cmake --build build -j

# Run the QE-comparison subset
cd build && ctest -R QEValidation --output-on-failure

# Full validation suite (all systems)
ctest -j2 --output-on-failure`}</code>
        </pre>
        <p style={{opacity: 0.85}}>
          Adding a new comparison? Add a system folder under <code>benchmarks/</code> with
          a matching <code>pw.in</code> and <code>kronos.yaml</code>, and a regression test
          under <code>test/test_validation.cpp::QEValidation.*</code>. PRs welcome.
        </p>
      </div>
    </section>
  );
}

export default function Benchmarks(): ReactNode {
  return (
    <Layout
      title="Benchmarks"
      description="KRONOS validation against Quantum ESPRESSO — Si LDA agreement to 0.07 meV/atom, forces match analytic + FD to 5 significant figures.">
      <HeadlineHero />
      <Methodology />
      <QEComparisonTable />
      <ChartSection />
      <ComponentBreakdown />
      <ForceValidation />
      <MultiSystemTable />
      <Reproducibility />
      <Citations />
    </Layout>
  );
}
