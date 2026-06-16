import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

// Three-tile feature grid below the hero. No SVG dependencies yet — those
// are scaffold-only illustrations we'll replace with real iconography later.
type FeatureItem = {
  title: string;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Validated against QE',
    description: (
      <>
        Total energies agree with Quantum ESPRESSO to <strong>0.07 meV/atom</strong> on
        Si LDA Γ-only and <strong>0.15 meV/atom</strong> on Si 4×4×4 shifted.
        Forces match to 5 significant figures via finite differences.
      </>
    ),
  },
  {
    title: 'GPU-accelerated',
    description: (
      <>
        Pluggable backends for <strong>CUDA</strong> (NVIDIA), <strong>HIP</strong> (AMD),
        and <strong>Metal</strong> (Apple Silicon, fp32 research tier). Physics code
        stays vendor-agnostic; adding a backend touches only one folder.
      </>
    ),
  },
  {
    title: 'MPI-parallel & PAW-ready',
    description: (
      <>
        k-point and band parallelism via MPI. PAW pseudopotentials, hybrid
        functionals (PBE0, HSE06), spin-polarized LSDA, and variable-cell
        relaxation — all in a single binary.
      </>
    ),
  },
];

function Feature({title, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
