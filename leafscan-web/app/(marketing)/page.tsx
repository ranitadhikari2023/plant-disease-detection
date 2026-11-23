import {
  ArrowRight,
  Cpu,
  Microscope,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import Link from "next/link";

export default function HomePage() {
  return (
    <div className="flex flex-col items-center">
      {/* Hero Section */}
      <section className="text-center py-10 sm:py-16 max-w-3xl">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-leaf-100/80 border border-leaf-200 text-leaf-800 text-xs font-mono font-medium mb-6">
          <Sparkles className="h-3.5 w-3.5" />
          <span>LeafScan V3 Production Engine Ready</span>
        </div>

        <h1 className="text-4xl sm:text-5xl lg:text-6xl font-serif font-extrabold tracking-tight text-bark-900 leading-[1.15]">
          Scientific Foliar Screening for Solanaceous Crops
        </h1>

        <p className="mt-5 text-base sm:text-lg text-bark-600 leading-relaxed max-w-2xl mx-auto">
          Two-tier hierarchical computer vision with CBAM attention,
          deterministic safety gating, and conformal prediction guarantees for
          Potato and Tomato pathology.
        </p>

        <div className="mt-8 flex flex-wrap gap-3.5 justify-center">
          <Link
            href="/diagnose"
            className="inline-flex items-center gap-2 rounded-xl bg-leaf-600 px-6 py-3.5 text-base font-semibold text-white shadow-float hover:bg-leaf-700 active:scale-98 transition-all min-h-[48px]"
          >
            Launch Field Diagnostic
            <ArrowRight className="h-4 w-4" />
          </Link>

          <Link
            href="/model"
            className="inline-flex items-center gap-2 rounded-xl border border-bark-300 bg-white px-6 py-3.5 text-base font-semibold text-bark-900 shadow-xs hover:bg-bark-50 active:scale-98 transition-all min-h-[48px]"
          >
            Inspect Model Manifest
          </Link>
        </div>
      </section>

      {/* Feature Grid */}
      <section className="w-full py-10 border-t border-bark-200/60">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="rounded-2xl border border-bark-200/70 bg-white p-6 shadow-xs">
            <div className="rounded-xl bg-leaf-50 p-3 text-leaf-700 w-fit mb-4">
              <Microscope className="h-6 w-6" />
            </div>
            <h3 className="text-lg font-serif font-bold text-bark-900">
              Two-Tier Hierarchical Inference
            </h3>
            <p className="mt-2 text-xs text-bark-600 leading-relaxed">
              Tier 1 botanist classifies host species (Potato vs Tomato) with a
              90% confidence requirement. Only verified species advance to
              crop-specific CBAM attention specialists.
            </p>
          </div>

          <div className="rounded-2xl border border-bark-200/70 bg-white p-6 shadow-xs">
            <div className="rounded-xl bg-teal-500/10 p-3 text-teal-700 w-fit mb-4">
              <ShieldCheck className="h-6 w-6" />
            </div>
            <h3 className="text-lg font-serif font-bold text-bark-900">
              Deterministic Safety Authority
            </h3>
            <p className="mt-2 text-xs text-bark-600 leading-relaxed">
              Machine learning models never independently recommend treatments.
              High-consequence pathogens like Late Blight trigger non-negotiable
              safety escalation banners.
            </p>
          </div>

          <div className="rounded-2xl border border-bark-200/70 bg-white p-6 shadow-xs">
            <div className="rounded-xl bg-amber-500/10 p-3 text-amber-700 w-fit mb-4">
              <Cpu className="h-6 w-6" />
            </div>
            <h3 className="text-lg font-serif font-bold text-bark-900">
              90% Conformal Prediction Sets
            </h3>
            <p className="mt-2 text-xs text-bark-600 leading-relaxed">
              Provides finite-sample statistical guarantees that the true
              pathology is covered within the returned candidate set, avoiding
              misleading point-predictions.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
