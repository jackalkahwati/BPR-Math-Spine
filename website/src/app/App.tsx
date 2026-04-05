import { Hero } from './components/Hero';
import { WhatIsBPR } from './components/WhatIsBPR';
import { TheoryInBrief } from './components/TheoryInBrief';
import { PhysicsLandscape } from './components/PhysicsLandscape';
import { HowItWorks } from './components/HowItWorks';
import { PredictionFlow } from './components/PredictionFlow';
import { ResonantPrimeSubstrate } from './components/ResonantPrimeSubstrate';
import { WhySection } from './components/WhySection';
import { WhatsInTheBox } from './components/WhatsInTheBox';
import { ForVisitors } from './components/ForVisitors';
import { HowToTry } from './components/HowToTry';
import { FAQ } from './components/FAQ';
import { Footer } from './components/Footer';

export default function App() {
  return (
    <div className="min-h-screen w-full">
      <Hero />
      <WhatIsBPR />
      <TheoryInBrief />
      <PhysicsLandscape />
      <HowItWorks />
      <PredictionFlow />
      <ResonantPrimeSubstrate />
      <WhySection />
      <WhatsInTheBox />
      <ForVisitors />
      <HowToTry />
      <FAQ />
      <Footer />
    </div>
  );
}
