import type { Metadata } from "next";
import { CompareView } from "@/components/data/CompareView";

export const metadata: Metadata = {
  title: "Compare versions",
  description: "Switch between the deterministic mix, the learned versions and the shuffled control of a data track at the same moment, and rate each one.",
};

export default function ComparePage() {
  return <CompareView />;
}
