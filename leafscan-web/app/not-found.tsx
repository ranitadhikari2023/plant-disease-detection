import { ArrowLeft, Leaf } from "lucide-react";
import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[50vh] text-center px-4">
      <div className="rounded-full bg-leaf-100 p-4 text-leaf-700 mb-4">
        <Leaf className="h-10 w-10" />
      </div>
      <h2 className="text-3xl font-serif font-bold text-bark-900">
        404 — Page Not Found
      </h2>
      <p className="mt-2 max-w-md text-sm text-bark-500">
        The requested field diagnostic view or resource does not exist.
      </p>
      <div className="mt-6">
        <Link
          href="/diagnose"
          className="inline-flex items-center gap-2 rounded-xl bg-leaf-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-leaf-700"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Foliar Scanner
        </Link>
      </div>
    </div>
  );
}
