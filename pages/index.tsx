import Head from 'next/head';
import FaceMatch from '../components/FaceMatch';

export default function Home() {
  return (
    <>
      <Head>
        <title>FaceMatch AI — Forensic Facial Recognition</title>
        <meta name="description" content="Advanced AI-powered facial recognition and comparison using deep learning" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🔬</text></svg>" />
        <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet" />
      </Head>
      <FaceMatch />
    </>
  );
}
